import os
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ROOT: ClassVar[Path] = Path(__file__).resolve().parents[2]
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # MySQL
    mysql_host: str = Field(default_factory=lambda: os.getenv("MYSQL_HOST", ""))
    mysql_port: int = Field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_user: str = Field(default_factory=lambda: os.getenv("MYSQL_USER", ""))
    mysql_password: str = Field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    mysql_db: str = Field(default_factory=lambda: os.getenv("MYSQL_DB", ""))

    # Seq2Seq
    seq2seq_model_name: str = "google/flan-t5-base"
    seq2seq_max_new_tokens: int = 192
    seq2seq_temperature: float = 0.0

    # App
    app_env: str = "dev"
    version: str = "5.5.0"
    # Hybrid pipeline toggles
    enable_implicit: bool = True
    enable_llm_verifier: bool = True

    # protonet integration
    protonet_mode: str = Field(default="import")  # import | http
    protonet_url: str = Field(default="http://127.0.0.1:8011")
    implicit_min_confidence: float = 0.08

    # LLM verifier
    llm_provider: str = "groq"  # ollama_openai | openai_compatible
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
        or os.getenv("GROQ_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.groq.com/openai/v1"
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    llm_model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME")
        or os.getenv("GROQ_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "llama-3.1-8b-instant"
    )
    llm_timeout_seconds: int = 60

    # Output control
    max_implicit_candidates: int = 5
    max_verified_predictions: int = 8
        

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
