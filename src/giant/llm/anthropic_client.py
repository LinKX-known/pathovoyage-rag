"""调用对应模型接口，解析工具动作并记录用量与费用。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from aiolimiter import AsyncLimiter
from anthropic import APIConnectionError, AsyncAnthropic, RateLimitError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from giant.config import Settings, settings
from giant.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from giant.llm.converters import (
    count_image_pixels_in_messages,
    get_system_prompt_for_anthropic,
    messages_to_anthropic,
)
from giant.llm.model_registry import DEFAULT_ANTHROPIC_MODEL, validate_model_id
from giant.llm.pricing import calculate_cost, calculate_image_cost_anthropic
from giant.llm.protocol import (
    LLMError,
    LLMParseError,
    LLMResponse,
    Message,
    StepResponse,
    TokenUsage,
)
from giant.llm.schemas import step_response_json_schema
from giant.utils.logging import get_logger

logger = get_logger(__name__)


def _build_submit_step_tool() -> dict[str, Any]:

    return {
        "name": "submit_step",
        "description": (
            "Submit your reasoning and action for this navigation step. "
            "You MUST call this tool to provide your response."
        ),
        "input_schema": step_response_json_schema(),
    }


def _parse_tool_use_to_step_response(tool_input: dict[str, Any]) -> StepResponse:

    try:


        if isinstance(tool_input.get("action"), str):
            action_str = tool_input["action"]
            try:
                tool_input = {
                    **tool_input,
                    "action": json.loads(action_str),
                }
            except json.JSONDecodeError as e:
                raise LLMParseError(
                    "Anthropic tool_input.action was a string but was not "
                    f"valid JSON: {e}",
                    raw_output=str(tool_input),
                    provider="anthropic",
                ) from e

        return StepResponse.model_validate(tool_input)
    except ValidationError as e:
        raise LLMParseError(
            f"Failed to parse StepResponse: {e}",
            raw_output=str(tool_input),
            provider="anthropic",
        ) from e


@dataclass
class AnthropicProvider:
    """封装对应模型的消息格式、重试、动作解析和计费。"""

    model: str = DEFAULT_ANTHROPIC_MODEL
    settings: Settings = field(default_factory=lambda: settings)

    # 内部状态
    _client: AsyncAnthropic = field(init=False, repr=False)
    _limiter: AsyncLimiter = field(init=False, repr=False)
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False)

    def __post_init__(self) -> None:

        validate_model_id(self.model, provider="anthropic")
        api_key = self.settings.require_anthropic_key()
        self._client = AsyncAnthropic(api_key=api_key)
        self._limiter = AsyncLimiter(
            max_rate=self.settings.ANTHROPIC_RPM,
            time_period=60,
        )
        self._circuit_breaker = CircuitBreaker(
            provider_name="anthropic",
            config=CircuitBreakerConfig(
                failure_threshold=10,
                cooldown_seconds=60.0,
            ),
        )

    def get_model_name(self) -> str:

        return self.model

    def get_provider_name(self) -> str | None:
        return "anthropic"

    def get_target_size(self) -> int:

        return self.settings.IMAGE_SIZE_ANTHROPIC

    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        """发送多模态消息并解析模型的工具动作。"""
        self._circuit_breaker.check()

        try:
            return await self._call_with_retry(messages)
        except LLMParseError:
            # 模式与提示词不匹配时，不应视为服务提供方故障。
            raise
        except (RateLimitError, APIConnectionError) as e:
            self._circuit_breaker.record_failure()
            raise LLMError(
                f"Anthropic API call failed after retries: {e}",
                provider="anthropic",
                model=self.model,
                cause=e,
            ) from e
        # 注意：其他异常（如应用错误、校验错误）
        # 应直接向上传播，不触发熔断；只有临时性或远端
        # 错误才应影响熔断器状态。


    # - 最短等待 1 秒，避免临时错误时密集重试。
    # - 最长等待 60 秒，避免退避时间过长。
    # - 最多尝试 6 次；指数退避总耗时约 2～3 分钟。
    # 这些值符合生产环境建议，暂不开放配置。
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def _call_with_retry(self, messages: list[Message]) -> LLMResponse:

        try:
            # 构建请求。
            system_prompt = get_system_prompt_for_anthropic(messages)
            anthropic_messages = messages_to_anthropic(messages)


            async with self._limiter:
                start_time = time.perf_counter()
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=self.settings.ANTHROPIC_TEMPERATURE,
                    system=system_prompt or "",
                    messages=anthropic_messages,
                    tools=[_build_submit_step_tool()],
                    tool_choice={"type": "tool", "name": "submit_step"},
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # 提前计算用量和费用，以便解析失败时仍能记录。
            usage = response.usage
            if usage is None:
                raise LLMError(
                    "API response missing usage data - cannot track costs",
                    provider="anthropic",
                    model=self.model,
                )
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens

            if prompt_tokens is None or completion_tokens is None:
                raise LLMError(
                    "API response has None token counts "
                    f"(input={prompt_tokens}, output={completion_tokens})",
                    provider="anthropic",
                    model=self.model,
                )

            total_tokens = prompt_tokens + completion_tokens

            text_cost = calculate_cost(self.model, prompt_tokens, completion_tokens)
            image_pixels = count_image_pixels_in_messages(messages)
            image_cost = calculate_image_cost_anthropic(self.model, image_pixels)
            total_cost = text_cost + image_cost

            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=total_cost,
            )

            # 在响应中查找工具调用。
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_step":
                    tool_use_block = block
                    break

            if tool_use_block is None:
                raise LLMParseError(
                    "No submit_step tool use in response",
                    raw_output=str(response.content),
                    provider="anthropic",
                    model=self.model,
                    usage=token_usage,
                )

            tool_input = tool_use_block.input
            if not isinstance(tool_input, dict):
                raise LLMParseError(
                    "submit_step tool input missing or not an object",
                    raw_output=str(tool_input),
                    provider="anthropic",
                    model=self.model,
                    usage=token_usage,
                )

            try:
                step_response = _parse_tool_use_to_step_response(tool_input)
            except LLMParseError as e:
                raise LLMParseError(
                    str(e),
                    raw_output=e.raw_output,
                    provider=e.provider or "anthropic",
                    model=e.model or self.model,
                    usage=token_usage,
                ) from e

            self._circuit_breaker.record_success()

            return LLMResponse(
                step_response=step_response,
                usage=token_usage,
                model=self.model,
                latency_ms=latency_ms,
            )

        except (RateLimitError, APIConnectionError):

            raise
        except LLMParseError:
            # 不要再次包装解析错误。
            raise
        except LLMError:

            raise
        except Exception as e:
            raise LLMError(
                f"Anthropic API call failed: {e}",
                provider="anthropic",
                model=self.model,
                cause=e,
            ) from e
