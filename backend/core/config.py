# proto/backend/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "12345"
    mysql_db: str = "protodb"

    # Seq2Seq
    seq2seq_model_name: str = "google/flan-t5-small"
    seq2seq_max_new_tokens: int = 192
    seq2seq_temperature: float = 0.0

    # App
    app_env: str = "dev"
    # Hybrid pipeline toggles
    enable_implicit: bool = True
    enable_llm_verifier: bool = True

    # ProtoBackend integration
    protobackend_mode: str = "import"   # import | http
    protobackend_url: str = "http://127.0.0.1:8011"
    implicit_min_confidence: float = 0.45

    # LLM verifier
    llm_provider: str = "groq"  # ollama_openai | openai_compatible
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = ""
    llm_model_name: str = "llama3.1:8b"
    llm_timeout_seconds: int = 60

    # Output control
    max_implicit_candidates: int = 5
    max_verified_predictions: int = 8
        

    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = False

    @property
    def mysql_url(self) -> str:
        # SQLAlchemy URL
        # mysql+pymysql://user:pass@host:port/dbname?charset=utf8mb4
        user = self.mysql_user
        pw = self.mysql_password
        host = self.mysql_host
        port = self.mysql_port
        db = self.mysql_db
        return f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4"


settings = Settings()

