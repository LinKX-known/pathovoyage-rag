"""登记可用模型及默认模型，并校验模型名称。"""

from __future__ import annotations

DEFAULT_OPENAI_MODEL: str = "gpt-5.2"
DEFAULT_ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"
DEFAULT_GOOGLE_MODEL: str = "gemini-3-pro-preview"
DEFAULT_QWEN_MODEL: str = "qwen-vl-max"
DEFAULT_MODELS_BY_PROVIDER: dict[str, str] = {
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "google": DEFAULT_GOOGLE_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
}

OPENAI_MODELS: frozenset[str] = frozenset(
    {
        DEFAULT_OPENAI_MODEL,
    }
)

ANTHROPIC_MODELS: frozenset[str] = frozenset(
    {
        DEFAULT_ANTHROPIC_MODEL,
    }
)

GOOGLE_MODELS: frozenset[str] = frozenset(
    {
        DEFAULT_GOOGLE_MODEL,
    }
)

QWEN_MODELS: frozenset[str] = frozenset(
    {
        DEFAULT_QWEN_MODEL,
        "qwen-vl-plus",
        # 新一代文本+视觉模型：能力更强、上下文 1M，单价按百炼价目表折算更低。
        # 注意 ID 写法是 qwen3.8-max（不是 qwen-3.8-max），写错会 404。
        "qwen3.8-max",
        "qwen3.7-plus",
        "qwen3-vl-plus",
    }
)

APPROVED_MODELS: frozenset[str] = (
    OPENAI_MODELS | ANTHROPIC_MODELS | GOOGLE_MODELS | QWEN_MODELS
)

APPROVED_MODELS_BY_PROVIDER: dict[str, frozenset[str]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "google": GOOGLE_MODELS,
    "qwen": QWEN_MODELS,
}


def get_default_model(provider: str) -> str:
    """返回指定服务提供方的默认模型名称。"""
    try:
        return DEFAULT_MODELS_BY_PROVIDER[provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported providers: "
            + ", ".join(sorted(DEFAULT_MODELS_BY_PROVIDER))
        ) from e


def validate_model_id(model: str, *, provider: str | None = None) -> None:
    """校验模型名称是否属于指定服务提供方。"""
    if provider is None:
        allowed = APPROVED_MODELS
        label = "GIANT approved models"
    else:
        if provider not in APPROVED_MODELS_BY_PROVIDER:
            raise ValueError(
                f"Unknown provider: {provider!r}. Supported providers: "
                + ", ".join(sorted(APPROVED_MODELS_BY_PROVIDER))
            )
        allowed = APPROVED_MODELS_BY_PROVIDER[provider]
        label = f"{provider} approved models"

    if model not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"Model {model!r} is not approved. Allowed for {label}: {allowed_list}. "
            "See docs/models/model-registry.md."
        )
