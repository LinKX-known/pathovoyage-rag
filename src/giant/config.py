"""读取环境变量和运行参数，校验模型密钥并解析提示词覆盖配置。"""

from pathlib import Path
from typing import TypeGuard

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_SECRET_MARKERS: tuple[str, ...] = (
    "your-key",
    "changeme",
)


class ConfigError(Exception):
    """配置缺失或参数非法时抛出的错误。"""

    def __init__(self, key_name: str, env_var: str) -> None:

        self.key_name = key_name
        self.env_var = env_var
        message = (
            f"{key_name} not configured. "
            f"Set it in .env file or {env_var} environment variable."
        )
        super().__init__(message)


class Settings(BaseSettings):
    """从环境变量加载模型、裁剪、限速与预算参数。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    DASHSCOPE_API_KEY: str | None = None
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    HUGGINGFACE_TOKEN: str | None = None


    PATHO_R1_BASE_URL: str = "http://localhost:18000/v1"
    PATHO_R1_MODEL: str = "patho-r1-7b"

    # 知识检索（RAG）：默认关闭，开启后把教科书/判据证据注入规划与仲裁环节。
    RAG_ENABLED: bool = False
    RAG_CORPUS_PATH: str = "data/corpus/textbooks_colorectal.jsonl"
    RAG_SPEC_PATH: str = "data/corpus/sources.json"
    RAG_RAW_DIR: str = "data/rag_raw"
    RAG_TOP_K: int = 6
    RAG_MIN_SCORE: float = 0.0
    RAG_MIN_COVERAGE: float = 0.0
    RAG_RELATIVE_CUTOFF: float = 0.25

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console"


    WSI_LONG_SIDE_TARGET: int = 1000
    MAX_ITERATIONS: int = 20
    OVERSAMPLING_BIAS: float = 0.85  # 对较大裁剪尺寸的偏置。
    THUMBNAIL_SIZE: int = 1024  # 送入多模态模型上下文的缩略图长边尺寸。

    # 图像生成
    JPEG_QUALITY: int = 85


    IMAGE_SIZE_OPENAI: int = 1000
    IMAGE_SIZE_ANTHROPIC: int = 500
    IMAGE_SIZE_QWEN: int = 1000

    # 请求限速
    OPENAI_RPM: int = 60
    ANTHROPIC_RPM: int = 60
    QWEN_RPM: int = 60

    # Qwen3 系列的思考模式开关：默认关闭——思考链会污染 JSON 输出并让输出 token 翻倍。
    QWEN_ENABLE_THINKING: bool = False

    # 大语言模型采样参数。
    OPENAI_TEMPERATURE: float = 0.0
    ANTHROPIC_TEMPERATURE: float = 0.0
    QWEN_TEMPERATURE: float = 0.0

    # 费用预算限制
    DEFAULT_BUDGET_USD: float = 0.0  # 0 表示不限制预算。

    # 提示词覆盖设置（用于复现论文或实验）。


    GIANT_SYSTEM_PROMPT: str | None = None
    GIANT_SYSTEM_PROMPT_PATH: str | None = None
    GIANT_SYSTEM_PROMPT_OPENAI: str | None = None
    GIANT_SYSTEM_PROMPT_OPENAI_PATH: str | None = None
    GIANT_SYSTEM_PROMPT_ANTHROPIC: str | None = None
    GIANT_SYSTEM_PROMPT_ANTHROPIC_PATH: str | None = None

    @model_validator(mode="after")
    def _apply_paper_parameter_overrides(self) -> "Settings":
        if "WSI_LONG_SIDE_TARGET" in self.model_fields_set:
            if "IMAGE_SIZE_OPENAI" not in self.model_fields_set:
                self.IMAGE_SIZE_OPENAI = self.WSI_LONG_SIDE_TARGET
            if "IMAGE_SIZE_ANTHROPIC" not in self.model_fields_set:
                self.IMAGE_SIZE_ANTHROPIC = self.WSI_LONG_SIDE_TARGET
        return self

    @staticmethod
    def _is_configured_secret(value: str | None) -> TypeGuard[str]:
        if value is None:
            return False

        stripped = value.strip()
        if stripped == "":
            return False

        lowered = stripped.lower()
        return not any(marker in lowered for marker in _PLACEHOLDER_SECRET_MARKERS)

    def require_openai_key(self) -> str:

        if not self._is_configured_secret(self.OPENAI_API_KEY):
            raise ConfigError("OpenAI API key", "OPENAI_API_KEY")
        return self.OPENAI_API_KEY

    def require_anthropic_key(self) -> str:

        if not self._is_configured_secret(self.ANTHROPIC_API_KEY):
            raise ConfigError("Anthropic API key", "ANTHROPIC_API_KEY")
        return self.ANTHROPIC_API_KEY

    def require_dashscope_key(self) -> str:

        if not self._is_configured_secret(self.DASHSCOPE_API_KEY):
            raise ConfigError("DashScope API key", "DASHSCOPE_API_KEY")
        return self.DASHSCOPE_API_KEY

    def require_google_key(self) -> str:

        if not self._is_configured_secret(self.GOOGLE_API_KEY):
            raise ConfigError("Google API key", "GOOGLE_API_KEY")
        return self.GOOGLE_API_KEY

    def require_huggingface_token(self) -> str:

        if not self._is_configured_secret(self.HUGGINGFACE_TOKEN):
            raise ConfigError("HuggingFace token", "HUGGINGFACE_TOKEN")
        return self.HUGGINGFACE_TOKEN

    @staticmethod
    def _read_prompt_file(path_str: str) -> str:
        try:
            return Path(path_str).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(
                f"Failed to read prompt file: {path_str!r}. "
                "Ensure the path exists and is readable."
            ) from e

    def get_giant_system_prompt(self, *, provider: str | None) -> str | None:
        """优先读取文件或环境变量覆盖值，未配置时使用内置系统提示词。"""
        provider_norm = provider.strip().lower() if provider else None

        candidates: list[tuple[str | None, str | None]] = []
        if provider_norm == "openai":
            candidates.append(
                (self.GIANT_SYSTEM_PROMPT_OPENAI_PATH, self.GIANT_SYSTEM_PROMPT_OPENAI)
            )
        elif provider_norm == "anthropic":
            candidates.append(
                (
                    self.GIANT_SYSTEM_PROMPT_ANTHROPIC_PATH,
                    self.GIANT_SYSTEM_PROMPT_ANTHROPIC,
                )
            )

        candidates.append((self.GIANT_SYSTEM_PROMPT_PATH, self.GIANT_SYSTEM_PROMPT))

        for path_str, prompt in candidates:
            if path_str:
                return self._read_prompt_file(path_str)
            if prompt:
                return prompt
        return None


# 提供单例实例，方便导入使用。
settings = Settings()
