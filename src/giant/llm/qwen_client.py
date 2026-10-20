"""调用千问视觉语言模型，解析文本动作并统计调用费用。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

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
from giant.llm.converters import get_system_prompt_for_qwen, messages_to_qwen
from giant.llm.model_registry import DEFAULT_QWEN_MODEL, validate_model_id
from giant.llm.pricing import calculate_cost
from giant.llm.protocol import (
    LLMError,
    LLMParseError,
    LLMResponse,
    Message,
    StepResponse,
    TokenUsage,
)
from giant.utils.logging import get_logger

logger = get_logger(__name__)

# 千问不支持强制结构化输出，因此需要通过提示词要求它

_JSON_OUTPUT_INSTRUCTION = """OUTPUT FORMAT (MANDATORY):
Respond with ONLY a single valid JSON object. No markdown fences, no extra text.
The JSON must have exactly two keys: "reasoning" and "action".

To run PLIP pre-screening (only as your very first action, if required), output:
{"reasoning": "<reasoning>", "action": {"action_type": "select", "candidates": []}}

To zoom into a region, output:
{"reasoning": "<step-by-step reasoning>", "action": {"action_type": "crop", "x": <int>, "y": <int>, "width": <int>, "height": <int>}}
where x, y, width, height are Level-0 pixel coordinates.

To give the final answer, output:
{"reasoning": "<reasoning>", "action": {"action_type": "answer", "answer_text": "<full answer addressing ALL parts of the question: presence of tumor, histologic type and grade, AND key morphologic features at architectural and cytologic levels>", "report": {"diagnosis": "<diagnosis>", "confidence": "<high|medium|low>", "evidence": ["<supporting feature>"], "differential": ["<alternative diagnosis>"]}}}
"answer_text" must answer EVERY part of the question completely, not just the diagnosis. The "report" field is optional but recommended."""


def _strip_markdown_fences(text: str) -> str:

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.lstrip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rstrip("`").strip()
    return stripped


def _trim_to_json(text: str) -> str:
    """把 JSON 之前的散文/思考链剥掉，返回从第一个 `{` 开始的文本。

    Qwen3 系列在 thinking 模式下会在 JSON 前输出分析段落；模型的
    ``response_format=json_object`` 并不能保证绝对没有前缀，所以解析前统一裁剪。
    """
    start = text.find("{")
    return text[start:] if start > 0 else text


def _extra_body(model: str, *, enable_thinking: bool) -> dict[str, object]:
    """Qwen3 系列支持思考模式开关；更早的模型不接受该参数，按模型前缀决定是否下发。"""
    if model.strip().lower().startswith("qwen3"):
        return {"enable_thinking": enable_thinking}
    return {}


def _parse_qwen_step_response(
    *,
    output_text: str | None,
    model: str,
    token_usage: TokenUsage,
) -> StepResponse:

    if output_text is None or not output_text.strip():
        raise LLMParseError(
            "Empty output text in response",
            provider="qwen",
            model=model,
            usage=token_usage,
        )

    text = _trim_to_json(_strip_markdown_fences(output_text))

    try:

        decoder = json.JSONDecoder()
        raw_data, _ = decoder.raw_decode(text)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"Failed to parse JSON: {e}",
            raw_output=output_text,
            provider="qwen",
            model=model,
            usage=token_usage,
        ) from e


    if isinstance(raw_data, dict) and raw_data.get("action") == "select":
        raw_data["action"] = {"action_type": "select", "candidates": []}

    try:
        return StepResponse.model_validate(raw_data)
    except ValidationError as e:
        raise LLMParseError(
            f"Failed to parse StepResponse: {e}",
            raw_output=output_text,
            provider="qwen",
            model=model,
            usage=token_usage,
        ) from e


@dataclass
class QwenProvider:
    """封装千问的文本与图像调用、动作解析和计费。"""

    model: str = DEFAULT_QWEN_MODEL
    settings: Settings = field(default_factory=lambda: settings)


    _client: AsyncOpenAI = field(init=False, repr=False)
    _limiter: AsyncLimiter = field(init=False, repr=False)
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False)

    def __post_init__(self) -> None:

        validate_model_id(self.model, provider="qwen")
        api_key = self.settings.require_dashscope_key()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.DASHSCOPE_BASE_URL,
        )
        self._limiter = AsyncLimiter(
            max_rate=self.settings.QWEN_RPM,
            time_period=60,
        )
        self._circuit_breaker = CircuitBreaker(
            provider_name="qwen",
            config=CircuitBreakerConfig(
                failure_threshold=10,
                cooldown_seconds=60.0,
            ),
        )

    def get_model_name(self) -> str:

        return self.model

    def get_provider_name(self) -> str | None:
        return "qwen"

    def get_target_size(self) -> int:

        return self.settings.IMAGE_SIZE_QWEN

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 256,
    ) -> str:

        self._circuit_breaker.check()
        try:
            return await self._call_text_with_retry(
                system=system,
                user=user,
                max_tokens=max_tokens,
            )
        except (RateLimitError, APIConnectionError) as e:
            self._circuit_breaker.record_failure()
            raise LLMError(
                f"Qwen text completion failed after retries: {e}",
                provider="qwen",
                model=self.model,
                cause=e,
            ) from e

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def _call_text_with_retry(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
    ) -> str:

        try:
            async with self._limiter:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.settings.QWEN_TEMPERATURE,
                    max_tokens=max_tokens,
                    extra_body=_extra_body(
                        self.model,
                        enable_thinking=self.settings.QWEN_ENABLE_THINKING,
                    ),
                )
            content = response.choices[0].message.content if response.choices else None
            if content is None or not content.strip():
                raise LLMError(
                    "Empty completion content",
                    provider="qwen",
                    model=self.model,
                )
            self._circuit_breaker.record_success()
            return content.strip()
        except (RateLimitError, APIConnectionError):

            raise
        except LLMError:

            raise
        except Exception as e:
            raise LLMError(
                f"Qwen text completion failed: {e}",
                provider="qwen",
                model=self.model,
                cause=e,
            ) from e

    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        """发送导航消息并从模型文本中解析动作。"""
        self._circuit_breaker.check()

        try:
            return await self._call_with_retry(messages)
        except LLMParseError:
            # 模式与提示词不匹配时，不应视为服务提供方故障。
            raise
        except (RateLimitError, APIConnectionError) as e:
            self._circuit_breaker.record_failure()
            raise LLMError(
                f"Qwen API call failed after retries: {e}",
                provider="qwen",
                model=self.model,
                cause=e,
            ) from e

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def _call_with_retry(self, messages: list[Message]) -> LLMResponse:

        try:
            system_prompt = get_system_prompt_for_qwen(messages) or ""
            system_prompt = (system_prompt + "\n\n" + _JSON_OUTPUT_INSTRUCTION).strip()
            chat_messages = messages_to_qwen(messages)

            async with self._limiter:
                start_time = time.perf_counter()
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *chat_messages,
                    ],
                    temperature=self.settings.QWEN_TEMPERATURE,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                    extra_body=_extra_body(
                        self.model,
                        enable_thinking=self.settings.QWEN_ENABLE_THINKING,
                    ),
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            usage = response.usage
            if usage is None:
                raise LLMError(
                    "API response missing usage data - cannot track costs",
                    provider="qwen",
                    model=self.model,
                )

            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            total_tokens = prompt_tokens + completion_tokens


            cost_usd = calculate_cost(self.model, prompt_tokens, completion_tokens)
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )

            output_text = (
                response.choices[0].message.content if response.choices else None
            )
            step_response = _parse_qwen_step_response(
                output_text=output_text,
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
                f"Qwen API call failed: {e}",
                provider="qwen",
                model=self.model,
                cause=e,
            ) from e
