"""将统一消息结构转换为各模型接口需要的文本与图像格式。"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Any, Literal

from PIL import Image

from giant.llm.protocol import Message, MessageContent


def message_content_to_openai(
    content: MessageContent,
    *,
    role: Literal["user", "assistant"],
) -> dict[str, Any]:

    if content.type == "text":
        if content.text is None:
            raise ValueError("Text content requires 'text' field")
        return {
            "type": "output_text" if role == "assistant" else "input_text",
            "text": content.text,
        }
    elif content.type == "image":
        if role != "user":
            raise ValueError(
                "OpenAI Responses API only supports images in user messages"
            )
        if content.image_base64 is None:
            raise ValueError("Image content requires 'image_base64' field")
        return {
            "type": "input_image",
            "image_url": f"data:{content.media_type};base64,{content.image_base64}",
        }
    else:
        raise ValueError(f"Unknown content type: {content.type}")


def message_to_openai(message: Message) -> dict[str, Any]:


    # 系统消息使用不同的结构。
    if message.role == "system":
        # 系统消息通常只有文本。
        text_parts = [c.text for c in message.content if c.type == "text" and c.text]
        return {
            "role": "system",
            "content": "\n".join(text_parts),
        }

    return {
        "role": message.role,
        "content": [
            message_content_to_openai(c, role=message.role) for c in message.content
        ],
    }


def messages_to_openai_input(messages: list[Message]) -> list[dict[str, Any]]:
    """把内部消息列表转换为结构化接口的输入格式。"""
    result: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":

            continue
        result.append(message_to_openai(message))

    return result


def extract_system_prompt(messages: list[Message]) -> str | None:

    system_parts: list[str] = []
    for message in messages:
        if message.role == "system":
            for content in message.content:
                if content.type == "text" and content.text:
                    system_parts.append(content.text)

    return "\n".join(system_parts) if system_parts else None


def get_system_prompt_for_openai(messages: list[Message]) -> str | None:

    return extract_system_prompt(messages)


def message_content_to_anthropic(content: MessageContent) -> dict[str, Any]:

    if content.type == "text":
        if content.text is None:
            raise ValueError("Text content requires 'text' field")
        return {
            "type": "text",
            "text": content.text,
        }
    elif content.type == "image":
        if content.image_base64 is None:
            raise ValueError("Image content requires 'image_base64' field")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.image_base64,
            },
        }
    else:
        raise ValueError(f"Unknown content type: {content.type}")


def message_to_anthropic(message: Message) -> dict[str, Any]:

    if message.role == "system":
        raise ValueError(
            "System messages should be passed via 'system' parameter, "
            "not as messages. Use get_system_prompt_for_anthropic()."
        )

    return {
        "role": message.role,
        "content": [message_content_to_anthropic(c) for c in message.content],
    }


def messages_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """把内部消息列表转换为对应接口的消息格式。"""
    return [message_to_anthropic(m) for m in messages if m.role != "system"]


def get_system_prompt_for_anthropic(messages: list[Message]) -> str | None:

    return extract_system_prompt(messages)


def message_content_to_qwen(
    content: MessageContent,
    *,
    role: Literal["user", "assistant"],
) -> dict[str, Any]:

    if content.type == "text":
        if content.text is None:
            raise ValueError("Text content requires 'text' field")
        return {"type": "text", "text": content.text}
    elif content.type == "image":
        if role != "user":
            raise ValueError(
                "Qwen (Chat Completions) only supports images in user messages"
            )
        if content.image_base64 is None:
            raise ValueError("Image content requires 'image_base64' field")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{content.media_type};base64,{content.image_base64}"},
        }
    else:
        raise ValueError(f"Unknown content type: {content.type}")


def messages_to_qwen(messages: list[Message]) -> list[dict[str, Any]]:
    """把内部消息列表转换为千问接口的消息格式。"""
    result: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            continue
        if message.role == "assistant":
            text = "\n".join(
                c.text for c in message.content if c.type == "text" and c.text
            )
            result.append({"role": "assistant", "content": text})
        else:
            result.append(
                {
                    "role": "user",
                    "content": [
                        message_content_to_qwen(c, role="user") for c in message.content
                    ],
                }
            )

    return result


def get_system_prompt_for_qwen(messages: list[Message]) -> str | None:

    return extract_system_prompt(messages)


def count_images_in_messages(messages: list[Message]) -> int:

    count = 0
    for message in messages:
        for content in message.content:
            if content.type == "image":
                count += 1
    return count


def count_image_pixels_in_messages(messages: list[Message]) -> int:

    total_pixels = 0
    for message in messages:
        for content in message.content:
            if content.type != "image":
                continue
            if content.image_base64 is None:
                raise ValueError("Image content requires 'image_base64' field")
            if content.image_base64 == "":
                raise ValueError("Image content has empty 'image_base64' field")
            try:
                image_bytes = base64.b64decode(content.image_base64, validate=True)
            except binascii.Error as e:
                raise ValueError("Invalid base64 image data") from e
            if not image_bytes:
                raise ValueError("Image base64 decoded to empty bytes")
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
            total_pixels += width * height
    return total_pixels
