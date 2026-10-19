"""调用结构化模型接口，解析导航动作并记录用量与费用。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aiolimiter import AsyncLimiter
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
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
    count_images_in_messages,
    get_system_prompt_for_openai,
    messages_to_openai_input,
)
from giant.llm.model_registry import DEFAULT_OPENAI_MODEL, validate_model_id
from giant.llm.pricing import calculate_cost, calculate_image_cost_openai
from giant.llm.protocol import (
    LLMError,
    LLMParseError,
    LLMResponse,
    Message,
    StepResponse,
    TokenUsage,
)
from giant.llm.schemas import step_response_json_schema_openai
from giant.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# 类型协议
# =============================================================================


@runtime_checkable
class _OpenAIResponseUsage(Protocol):


    input_tokens: int
    output_tokens: int


# =============================================================================
# 辅助函数
# =============================================================================


def _build_json_schema() -> dict[str, Any]:

    return {
        "name": "StepResponse",
        "strict": True,
        "schema": step_response_json_schema_openai(),
    }


def _normalize_openai_response(data: dict[str, Any]) -> dict[str, Any]:

    if "action" not in data or not isinstance(data["action"], dict):
        return data

    action = data["action"]
    action_type = action.get("action_type")

    if action_type == "crop":
        normalized_action = {
            "action_type": "crop",
            "x": action.get("x"),
            "y": action.get("y"),
            "width": action.get("width"),
            "height": action.get("height"),
        }
    elif action_type == "answer":
        normalized_action = {
            "action_type": "answer",
            "answer_text": action.get("answer_text"),
            "report": action.get("report"),
        }
    elif action_type == "conch":
        normalized_action = {
            "action_type": "conch",
            "hypotheses": action.get("hypotheses"),
        }
    else:

        raise LLMParseError(
            (
                f"Unknown action_type '{action_type}'. Expected "
                "'crop', 'answer', or 'conch'."
            ),
            raw_output=str(action),
            provider="openai",
        )

    return {
        "reasoning": data.get("reasoning"),
        "action": normalized_action,
    }


def _calculate_openai_usage_and_cost(
    *,
    response_usage: _OpenAIResponseUsage | None,
    model: str,
    messages: list[Message],
) -> TokenUsage:
    if response_usage is None:
        raise LLMError(
            "API response missing usage data - cannot track costs",
            provider="openai",
            model=model,
        )
    prompt_tokens = response_usage.input_tokens
    completion_tokens = response_usage.output_tokens

    if prompt_tokens is None or completion_tokens is None:
        raise LLMError(
            "API response has None token counts "
            f"(input={prompt_tokens}, output={completion_tokens})",
            provider="openai",
            model=model,
        )

    total_tokens = prompt_tokens + completion_tokens

    text_cost = calculate_cost(model, prompt_tokens, completion_tokens)
    image_count = count_images_in_messages(messages)
    image_cost = calculate_image_cost_openai(model, image_count)
    total_cost = text_cost + image_cost

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=total_cost,
    )


def _parse_openai_step_response(
    *,
    output_text: str | None,
    model: str,
    token_usage: TokenUsage,
) -> StepResponse:
    if output_text is None:
        raise LLMParseError(
            "No output text in response",
            provider="openai",
            model=model,
            usage=token_usage,
        )
    if not output_text.strip():
        raise LLMParseError(
            "Empty output text in response",
            raw_output=output_text,
            provider="openai",
            model=model,
            usage=token_usage,
        )

    try:


        decoder = json.JSONDecoder()
        leading_ws = len(output_text) - len(output_text.lstrip())
        raw_data, end_idx = decoder.raw_decode(output_text, idx=leading_ws)
        if end_idx < len(output_text.rstrip()):
            logger.debug(
                "Ignored trailing text after JSON: %s",
                output_text[end_idx:].strip()[:50],
            )
        normalized_data = _normalize_openai_response(raw_data)
        return StepResponse.model_validate(normalized_data)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"Failed to parse JSON: {e}",
            raw_output=output_text,
            provider="openai",
            model=model,
            usage=token_usage,
        ) from e
    except ValidationError as e:
        raise LLMParseError(
            f"Failed to parse StepResponse: {e}",
            raw_output=output_text,
            provider="openai",
            model=model,
            usage=token_usage,
        ) from e
    except LLMParseError as e:
        raise LLMParseError(
            str(e),
            raw_output=e.raw_output,
            provider=e.provider or "openai",
            model=e.model or model,
            usage=token_usage,
        ) from e


@dataclass
class OpenAIProvider:
    """封装结构化模型调用、重试、动作解析和计费。"""

    model: str = DEFAULT_OPENAI_MODEL
    settings: Settings = field(default_factory=lambda: settings)


    _client: AsyncOpenAI = field(init=False, repr=False)
    _limiter: AsyncLimiter = field(init=False, repr=False)
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False)

    def __post_init__(self) -> None:

        validate_model_id(self.model, provider="openai")
        api_key = self.settings.require_openai_key()
        self._client = AsyncOpenAI(api_key=api_key)
        self._limiter = AsyncLimiter(
            max_rate=self.settings.OPENAI_RPM,
            time_period=60,
        )
        self._circuit_breaker = CircuitBreaker(
            provider_name="openai",
            config=CircuitBreakerConfig(
                failure_threshold=10,
                cooldown_seconds=60.0,
            ),
        )

    def get_model_name(self) -> str:

        return self.model

    def get_provider_name(self) -> str | None:
        return "openai"

    def get_target_size(self) -> int:

        return self.settings.IMAGE_SIZE_OPENAI

    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        """转换消息并调用模型，解析结构化导航动作。"""
        self._circuit_breaker.check()

        try:
            return await self._call_with_retry(messages)
        except LLMParseError:
            # 模式与提示词不匹配时，不应视为服务提供方故障。
            raise
        except (RateLimitError, APIConnectionError) as e:
            self._circuit_breaker.record_failure()
            raise LLMError(
                f"OpenAI API call failed after retries: {e}",
                provider="openai",
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
            system_prompt = get_system_prompt_for_openai(messages)
            input_messages = messages_to_openai_input(messages)


            json_schema = _build_json_schema()
            async with self._limiter:
                start_time = time.perf_counter()
                response = await self._client.responses.create(
                    model=self.model,
                    input=input_messages,
                    instructions=system_prompt,
                    temperature=self.settings.OPENAI_TEMPERATURE,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": json_schema["name"],
                            "schema": json_schema["schema"],
                            "strict": json_schema.get("strict", True),
                        }
                    },
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # 提前计算用量和费用，以便解析失败时仍能记录。
            token_usage = _calculate_openai_usage_and_cost(
                response_usage=response.usage,
                model=self.model,
                messages=messages,
            )
            step_response = _parse_openai_step_response(
                output_text=response.output_text,
                model=self.model,
                token_usage=token_usage,
            )

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
                f"OpenAI API call failed: {e}",
                provider="openai",
                model=self.model,
                cause=e,
            ) from e
