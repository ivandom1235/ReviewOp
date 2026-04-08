import os
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


class Settings(BaseSettings):
    ROOT: ClassVar[Path] = Path(__file__).resolve().parents[2]
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # MySQL
    mysql_host: str = Field(default_factory=lambda: os.getenv("MYSQL_HOST", ""))
    mysql_port: int = Field(default_factory=lambda: _env_int("MYSQL_PORT", 3306))
    mysql_user: str = Field(default_factory=lambda: os.getenv("MYSQL_USER", ""))
    mysql_password: str = Field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    mysql_db: str = Field(default_factory=lambda: os.getenv("MYSQL_DB", ""))

    # Seq2Seq
    seq2seq_model_name: str = Field(
        default="google/flan-t5-base",
        validation_alias=AliasChoices("REVIEWOP_SEQ2SEQ_MODEL_NAME", "SEQ2SEQ_MODEL_NAME"),
    )
    seq2seq_max_new_tokens: int = Field(
        default=192,
        validation_alias=AliasChoices("REVIEWOP_SEQ2SEQ_MAX_NEW_TOKENS", "SEQ2SEQ_MAX_NEW_TOKENS"),
    )
    seq2seq_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("REVIEWOP_SEQ2SEQ_TEMPERATURE", "SEQ2SEQ_TEMPERATURE"),
    )
    kg_embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("REVIEWOP_KG_EMBEDDING_MODEL_NAME", "KG_EMBEDDING_MODEL_NAME"),
    )
    open_aspect_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("REVIEWOP_OPEN_ASPECT_MODEL_NAME", "OPEN_ASPECT_MODEL_NAME"),
    )

    # App
    app_env: str = Field(default="dev", validation_alias=AliasChoices("REVIEWOP_ENV", "APP_ENV"))
    version: str = "6.0.0"
    # Hybrid pipeline toggles
    enable_implicit: bool = Field(default=True, validation_alias=AliasChoices("REVIEWOP_ENABLE_IMPLICIT", "ENABLE_IMPLICIT"))
    enable_llm_verifier: bool = Field(default=False, validation_alias=AliasChoices("REVIEWOP_ENABLE_LLM_VERIFIER", "ENABLE_LLM_VERIFIER"))

    # protonet integration
    protonet_mode: str = Field(default="import", validation_alias=AliasChoices("REVIEWOP_PROTONET_MODE", "PROTONET_MODE"))  # import | http
    protonet_url: str = Field(default="http://127.0.0.1:8011", validation_alias=AliasChoices("REVIEWOP_PROTONET_URL", "PROTONET_URL"))
    protonet_request_timeout_seconds: int = Field(
        default=30,
        validation_alias=AliasChoices("REVIEWOP_PROTONET_REQUEST_TIMEOUT_SECONDS", "PROTONET_REQUEST_TIMEOUT_SECONDS"),
    )
    protonet_bundle_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("REVIEWOP_PROTONET_BUNDLE_PATH", "PROTONET_BUNDLE_PATH"),
    )
    protonet_encoder_model: str = Field(
        default="microsoft/deberta-v3-base",
        validation_alias=AliasChoices("REVIEWOP_PROTONET_ENCODER_MODEL", "PROTONET_ENCODER_MODEL"),
    )
    implicit_min_confidence: float = Field(
        default=0.08,
        validation_alias=AliasChoices("REVIEWOP_IMPLICIT_MIN_CONFIDENCE", "IMPLICIT_MIN_CONFIDENCE"),
    )

    # LLM verifier
    llm_provider: str = Field(
        default="groq",
        validation_alias=AliasChoices("REVIEWOP_LLM_PROVIDER", "LLM_PROVIDER", "REVIEWOP_DEFAULT_LLM_PROVIDER", "DEFAULT_LLM_PROVIDER"),
    )  # ollama_openai | openai_compatible
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("REVIEWOP_LLM_BASE_URL", "LLM_BASE_URL", "GROQ_BASE_URL", "CLAUDE_BASE_URL", "OPENAI_BASE_URL", "OLLAMA_BASE_URL"),
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("REVIEWOP_LLM_API_KEY", "LLM_API_KEY", "GROQ_API_KEY", "CLAUDE_API_KEY", "OPENAI_API_KEY"),
    )
    llm_model_name: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("REVIEWOP_LLM_MODEL_NAME", "LLM_MODEL_NAME", "GROQ_MODEL", "CLAUDE_MODEL", "OPENAI_MODEL", "OLLAMA_MODEL"),
    )
    llm_timeout_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices("REVIEWOP_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"),
    )

    # Output control
    max_implicit_candidates: int = Field(
        default=5,
        validation_alias=AliasChoices("REVIEWOP_MAX_IMPLICIT_CANDIDATES", "MAX_IMPLICIT_CANDIDATES"),
    )
    max_verified_predictions: int = Field(
        default=8,
        validation_alias=AliasChoices("REVIEWOP_MAX_VERIFIED_PREDICTIONS", "MAX_VERIFIED_PREDICTIONS"),
    )

    @property
    def mysql_url(self) -> str:
        # SQLAlchemy URL
        # mysql+pymysql://user:pass@host:port/dbname?charset=utf8mb4
        user = self.mysql_user
        pw = self.mysql_password
        host = self.mysql_host
        port = self.mysql_port
        db = self.mysql_db
        if not all([user, pw, host, db]):
            raise RuntimeError("MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DB must be set")
        return f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4"

    @property
    def protonet_implicit_min_confidence(self) -> float:
        return self.implicit_min_confidence


settings = Settings()
