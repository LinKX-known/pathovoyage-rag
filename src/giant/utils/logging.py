"""配置结构化日志，并给日志条目附加切片与运行关联信息。"""

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any, cast

import structlog
from structlog.types import Processor

from giant.config import settings


_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_item_id: ContextVar[str | None] = ContextVar("item_id", default=None)
_step: ContextVar[int | None] = ContextVar("step", default=None)


def set_correlation_context(
    run_id: str | None = None,
    item_id: str | None = None,
    step: int | None = None,
) -> None:

    if run_id is not None:
        _run_id.set(run_id)
    if item_id is not None:
        _item_id.set(item_id)
    if step is not None:
        _step.set(step)


def clear_correlation_context() -> None:

    _run_id.set(None)
    _item_id.set(None)
    _step.set(None)


def _add_correlation_ids(
    logger: logging.Logger,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:

    _ = logger, method_name
    run_id = _run_id.get()
    item_id = _item_id.get()
    step = _step.get()

    if run_id is not None:
        event_dict["run_id"] = run_id
    if item_id is not None:
        event_dict["item_id"] = item_id
    if step is not None:
        event_dict["step"] = step

    return event_dict


def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
) -> None:

    level = level or settings.LOG_LEVEL
    log_format = log_format or settings.LOG_FORMAT

    # 各输出格式共用的处理器。
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_correlation_ids,
    ]

    if log_format == "json":

        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # 开发环境使用彩色控制台输出。
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 同步配置标准库日志。
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
