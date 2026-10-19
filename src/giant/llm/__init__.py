"""导出模型接口、响应结构，并按服务提供方创建客户端。"""

from giant.llm.anthropic_client import AnthropicProvider
from giant.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from giant.llm.model_registry import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QWEN_MODEL,
    validate_model_id,
)
from giant.llm.openai_client import OpenAIProvider
from giant.llm.qwen_client import QwenProvider
from giant.llm.pricing import (
    PRICING_USD_PER_1K,
    calculate_cost,
    calculate_image_cost_anthropic,
    calculate_image_cost_openai,
    calculate_total_cost,
    get_model_pricing,
)
from giant.llm.protocol import (
    Action,
    BoundingBoxAction,
    CircuitBreakerOpenError,
    ConchAction,
    DiagnosticReport,
    FinalAnswerAction,
    LLMError,
    LLMParseError,
    LLMProvider,
    LLMResponse,
    Message,
    MessageContent,
    StepResponse,
    TokenUsage,
)

__all__ = [
    "PRICING_USD_PER_1K",
    "Action",
    "AnthropicProvider",
    "BoundingBoxAction",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "ConchAction",
    "DiagnosticReport",
    "FinalAnswerAction",
    "LLMError",
    "LLMParseError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MessageContent",
    "OpenAIProvider",
    "QwenProvider",
    "StepResponse",
    "TokenUsage",
    "calculate_cost",
    "calculate_image_cost_anthropic",
    "calculate_image_cost_openai",
    "calculate_total_cost",
    "create_provider",
    "get_model_pricing",
]


def create_provider(
    provider: str,
    *,
    model: str | None = None,
) -> LLMProvider:
    """根据服务提供方名称和模型配置创建对应客户端。"""
    if provider == "openai":
        chosen_model = model or DEFAULT_OPENAI_MODEL
        validate_model_id(chosen_model, provider="openai")
        return OpenAIProvider(model=chosen_model)
    elif provider == "anthropic":
        chosen_model = model or DEFAULT_ANTHROPIC_MODEL
        validate_model_id(chosen_model, provider="anthropic")
        return AnthropicProvider(model=chosen_model)
    elif provider == "qwen":
        chosen_model = model or DEFAULT_QWEN_MODEL
        validate_model_id(chosen_model, provider="qwen")
        return QwenProvider(model=chosen_model)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported providers: 'openai', "
            "'anthropic', 'qwen'"
        )
