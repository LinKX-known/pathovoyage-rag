"""根据模型单价、文本用量和图像尺寸估算调用费用。"""

from __future__ import annotations

from typing import TypedDict

from giant.llm.model_registry import APPROVED_MODELS


class ModelPricing(TypedDict, total=False):
    """保存模型的文本与图像计费单价。"""

    input: float
    output: float
    image_base: float
    image_per_1k_px: float


PRICING_USD_PER_1K: dict[str, ModelPricing] = {

    "claude-sonnet-4-5-20250929": {
        "input": 0.003,
        "output": 0.015,
        "image_per_1k_px": 0.00048,
    },

    "gemini-3-pro-preview": {
        "input": 0.002,
        "output": 0.012,

    },

    "gpt-5.2": {
        "input": 0.00175,
        "output": 0.014,
        "image_base": 0.00255,
    },


    # 图像费用字段。
    "qwen-vl-max": {
        "input": 0.0029,
        "output": 0.0086,
    },
    "qwen-vl-plus": {
        "input": 0.0006,
        "output": 0.0017,
    },
    # 百炼价目表（中国北京区，USD/1M tokens 折算为 /1K）：
    # qwen3.8-max 输入 $1.65 / 输出 $4.951 per 1M → 0.00165 / 0.004951 per 1K
    "qwen3.8-max": {
        "input": 0.00165,
        "output": 0.004951,
    },
    # qwen3.7-plus 输入 $0.276 / 输出 $1.101 per 1M
    "qwen3.7-plus": {
        "input": 0.000276,
        "output": 0.001101,
    },
    # qwen3-vl-plus 与 qwen-vl-plus 同档，暂按后者计价（换区/调价时以官方价目表为准）
    "qwen3-vl-plus": {
        "input": 0.0006,
        "output": 0.0017,
    },
}

_APPROVED_MODELS_LIST = ", ".join(sorted(APPROVED_MODELS))


def get_model_pricing(model: str) -> ModelPricing:

    try:
        return PRICING_USD_PER_1K[model]
    except KeyError as e:
        raise ValueError(
            f"Unknown model {model!r}. Approved models: {_APPROVED_MODELS_LIST}. "
            "See docs/models/model-registry.md."
        ) from e


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:

    pricing = get_model_pricing(model)
    input_cost = prompt_tokens * pricing["input"] / 1000
    output_cost = completion_tokens * pricing["output"] / 1000
    return input_cost + output_cost


def calculate_image_cost_openai(model: str, image_count: int = 1) -> float:

    pricing = get_model_pricing(model)
    base_cost = pricing.get("image_base", 0.0)
    return base_cost * image_count


def calculate_image_cost_anthropic(
    model: str,
    image_pixels: int,
) -> float:

    pricing = get_model_pricing(model)
    per_1k_px = pricing.get("image_per_1k_px", 0.0)
    return (image_pixels / 1000) * per_1k_px


def calculate_total_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    image_count: int = 0,
    image_pixels: int = 0,
    provider: str = "openai",
) -> float:

    text_cost = calculate_cost(model, prompt_tokens, completion_tokens)

    if provider == "openai" and image_count > 0:
        return text_cost + calculate_image_cost_openai(model, image_count)
    elif provider == "anthropic" and image_pixels > 0:
        return text_cost + calculate_image_cost_anthropic(model, image_pixels)

    return text_cost
