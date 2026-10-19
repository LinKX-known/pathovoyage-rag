"""定义切片打开、区域读取失败时的异常类型和上下文信息。"""

from pathlib import Path


class WSIError(Exception):
    """切片操作失败的基础异常，可附带文件路径。"""

    def __init__(self, message: str, path: Path | str | None = None) -> None:

        self.path = Path(path) if path else None
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:

        if self.path:
            return f"{self.message} (path: {self.path})"
        return self.message


class WSIOpenError(WSIError):
    """切片文件无法打开时抛出。"""

    pass


class WSIReadError(WSIError):
    """切片区域或缩略图读取失败时抛出。"""

    def __init__(
        self,
        message: str,
        path: Path | str | None = None,
        *,
        level: int | None = None,
        location: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:

        self.level = level
        self.location = location
        self.size = size
        super().__init__(message, path)

    def _format_message(self) -> str:

        parts = [self.message]
        if self.path:
            parts.append(f"path={self.path}")
        if self.level is not None:
            parts.append(f"level={self.level}")
        if self.location is not None:
            parts.append(f"location={self.location}")
        if self.size is not None:
            parts.append(f"size={self.size}")

        if len(parts) == 1:
            return parts[0]
        return f"{parts[0]} ({', '.join(parts[1:])})"
