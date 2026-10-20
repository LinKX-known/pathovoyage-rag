"""Qwen 模型注册、计费与解析健壮性测试（离线）。"""

from __future__ import annotations

import pytest

from giant.llm.model_registry import validate_model_id
from giant.llm.pricing import calculate_cost
from giant.llm.protocol import TokenUsage
from giant.llm.qwen_client import (
    _extra_body,
    _parse_qwen_step_response,
    _trim_to_json,
)

_USAGE = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0)
_CROP_JSON = '{"reasoning": "r", "action": {"action_type": "crop", "x": 1, "y": 2, "width": 30, "height": 40}}'


# --------------------------------------------------------------------------- 解析健壮性


def test_trim_to_json_drops_thinking_preamble() -> None:
    text = "先看组织结构，再决定是否放大。\n" + _CROP_JSON
    assert _trim_to_json(text).startswith("{")


def test_trim_to_json_keeps_plain_json() -> None:
    assert _trim_to_json(_CROP_JSON) == _CROP_JSON


def test_parse_survives_thinking_preamble() -> None:
    """Qwen3 thinking 模式可能在 JSON 前带分析段落，解析器必须容忍。"""
    step = _parse_qwen_step_response(
        output_text="思路：先低倍看结构。\n" + _CROP_JSON,
        model="qwen3.8-max",
        token_usage=_USAGE,
    )
    assert step.action.action_type == "crop"
    assert step.action.width == 30


def test_parse_still_rejects_garbage() -> None:
    with pytest.raises(Exception):
        _parse_qwen_step_response(
            output_text="完全不是 JSON", model="qwen3.8-max", token_usage=_USAGE
        )


# --------------------------------------------------------------------------- 思考模式开关


def test_extra_body_sets_thinking_for_qwen3_only() -> None:
    assert _extra_body("qwen3.8-max", enable_thinking=False) == {"enable_thinking": False}
    assert _extra_body("qwen3.7-plus", enable_thinking=True) == {"enable_thinking": True}
    # 老模型不接受该参数，必须不下发
    assert _extra_body("qwen-vl-max", enable_thinking=False) == {}
    assert _extra_body("qwen-vl-plus", enable_thinking=False) == {}


# --------------------------------------------------------------------------- 白名单与计费


@pytest.mark.parametrize("model", ["qwen3.8-max", "qwen3.7-plus", "qwen3-vl-plus", "qwen-vl-max"])
def test_new_models_are_accepted(model: str) -> None:
    # 校验函数成功时返回 None，失败时抛 ValueError
    validate_model_id(model, provider="qwen")


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_model_id("qwen-3.8-max", provider="qwen")  # 错误写法不应通过


def test_qwen38_max_is_cheaper_than_qwen_vl_max() -> None:
    """同一 token 量下 qwen3.8-max 单价低于 qwen-vl-max（百炼北京区价目表折算）。"""
    new = calculate_cost("qwen3.8-max", 20_000, 3_000)
    old = calculate_cost("qwen-vl-max", 20_000, 3_000)
    assert new < old
    assert new == pytest.approx(0.0479, abs=1e-4)


def test_every_whitelisted_qwen_model_has_pricing() -> None:
    """白名单与价目表必须同步：否则计费时抛 ValueError 导致跑几轮崩掉。"""
    from giant.llm.model_registry import QWEN_MODELS

    for model in QWEN_MODELS:
        assert calculate_cost(model, 1000, 1000) > 0
