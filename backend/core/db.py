# proto/backend/core/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from core.config import settings


class Base(DeclarativeBase):
    pass


def ensure_database_exists() -> None:
    """
    Create the configured MySQL database if it does not exist yet.
    Requires valid MySQL host/user/password in settings.
    """
    url = make_url(settings.mysql_url)
    db_name = url.database
    if not db_name:
        raise RuntimeError("mysql_db is not configured")

    # Connect to MySQL server without selecting a specific database.
    admin_url = url.set(database=None)
    admin_engine = create_engine(
        admin_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )

    with admin_engine.begin() as conn:
        conn.execute(
            text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        )

    admin_engine.dispose()


ensure_database_exists()

engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
