"""生成导航动作的结构化输出模式。"""

from __future__ import annotations

from typing import Any


def step_response_json_schema() -> dict[str, Any]:

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reasoning for the action",
            },
            "action": {
                "description": "The action to take",
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action_type": {"type": "string", "enum": ["crop"]},
                            "x": {"type": "integer", "minimum": 0},
                            "y": {"type": "integer", "minimum": 0},
                            "width": {"type": "integer", "exclusiveMinimum": 0},
                            "height": {"type": "integer", "exclusiveMinimum": 0},
                        },
                        "required": ["action_type", "x", "y", "width", "height"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action_type": {"type": "string", "enum": ["answer"]},
                            "answer_text": {"type": "string", "minLength": 1},
                            "report": {
                                "type": ["object", "null"],
                                "description": "Optional structured diagnostic report",
                            },
                        },
                        "required": ["action_type", "answer_text"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action_type": {"type": "string", "enum": ["conch"]},
                            "hypotheses": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                                "description": "Text hypotheses to score with CONCH",
                            },
                        },
                        "required": ["action_type", "hypotheses"],
                    },
                ],
            },
        },
        "required": ["reasoning", "action"],
    }


def step_response_json_schema_openai() -> dict[str, Any]:

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": "Concise reasoning for the action",
            },
            "action": {
                "type": "object",
                "additionalProperties": False,
                "description": "Action: crop (x/y/width/height) or answer (text)",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["crop", "answer", "conch"],
                        "description": (
                            "crop=zoom region, answer=final response, "
                            "conch=score hypotheses"
                        ),
                    },
                    "x": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": "X coord (crop only, null for answer)",
                    },
                    "y": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": "Y coord (crop only, null for answer)",
                    },
                    "width": {
                        "type": ["integer", "null"],
                        "exclusiveMinimum": 0,
                        "description": "Width (crop only, null for answer)",
                    },
                    "height": {
                        "type": ["integer", "null"],
                        "exclusiveMinimum": 0,
                        "description": "Height (crop only, null for answer)",
                    },
                    "answer_text": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "description": "Final answer (answer only, null for crop)",
                    },
                    "hypotheses": {
                        "type": ["array", "null"],
                        "minItems": 1,
                        "description": (
                            "Text hypotheses to score (conch only, null otherwise)"
                        ),
                        "items": {"type": "string", "minLength": 1},
                    },
                    "report": {
                        "type": ["object", "null"],
                        "description": (
                            "Optional structured diagnostic report (answer only)"
                        ),
                    },
                },
                "required": [
                    "action_type",
                    "x",
                    "y",
                    "width",
                    "height",
                    "answer_text",
                    "hypotheses",
                    "report",
                ],
            },
        },
        "required": ["reasoning", "action"],
    }
