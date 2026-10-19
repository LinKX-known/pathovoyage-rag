"""在连续请求失败后暂停调用，并在冷却后试探恢复。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from giant.llm.protocol import CircuitBreakerOpenError
from giant.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """表示熔断器的关闭、断开或半开状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """设置失败阈值、冷却时间和半开试探次数。"""

    failure_threshold: int = 10
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class CircuitBreaker:
    """连续失败时阻止请求，并在冷却后逐步恢复。"""

    provider_name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # 内部状态
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_call_count: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def refresh_state(self, now: float | None = None) -> None:

        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return

        if now is None:
            now = time.monotonic()
        elapsed = now - self._last_failure_time
        if elapsed >= self.config.cooldown_seconds:
            self._transition_to_half_open()

    @property
    def is_closed(self) -> bool:

        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:

        return self.state == CircuitState.OPEN

    def _transition_to_half_open(self) -> None:

        logger.info(
            "Circuit breaker transitioning to half-open", provider=self.provider_name
        )
        self._state = CircuitState.HALF_OPEN
        self._half_open_call_count = 0
        self._success_count = 0

    def _transition_to_open(self) -> None:

        logger.warning(
            "Circuit breaker opening after failures",
            provider=self.provider_name,
            failure_count=self._failure_count,
        )
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()

    def _transition_to_closed(self) -> None:

        logger.info("Circuit breaker closing (recovered)", provider=self.provider_name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_call_count = 0

    def check(self) -> None:

        self.refresh_state()
        current_state = self.state

        if current_state == CircuitState.OPEN:
            cooldown_remaining = 0.0
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                cooldown_remaining = max(0.0, self.config.cooldown_seconds - elapsed)

            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for {self.provider_name}. "
                f"Too many consecutive failures. "
                f"Retry in {cooldown_remaining:.1f}s.",
                cooldown_remaining_seconds=cooldown_remaining,
                provider=self.provider_name,
            )

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_call_count >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker half-open call limit reached for "
                    f"{self.provider_name}. Waiting for pending calls to complete.",
                    cooldown_remaining_seconds=0.0,
                    provider=self.provider_name,
                )
            self._half_open_call_count += 1

    def record_success(self) -> None:

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # 成功后重置失败计数。
            self._failure_count = 0

    def record_failure(self) -> None:

        if self._state == CircuitState.HALF_OPEN:
            # 半开状态下的任何失败都会重新断开熔断器。
            logger.warning(
                "Failure in half-open state, reopening circuit",
                provider=self.provider_name,
            )
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def reset(self) -> None:

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_call_count = 0
        logger.info("Circuit breaker reset", provider=self.provider_name)
