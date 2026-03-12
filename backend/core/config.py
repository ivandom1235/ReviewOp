# proto/backend/core/config.py
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "H4ppyNY!"
    mysql_db: str = "protodb"

    # Seq2Seq
    seq2seq_model_name: str = "google/flan-t5-small"
    seq2seq_max_new_tokens: int = 192
    seq2seq_temperature: float = 0.0

    # App
    app_env: str = "dev"

    class Config:
        env_file = ".env"
        extra = "ignore"

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