from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.config import settings


class Base(DeclarativeBase):
    pass


def build_engine():
    url = make_url(settings.database_url)
    kwargs: dict[str, object] = {"pool_pre_ping": True, "echo": False}

    if url.drivername.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    elif url.drivername.startswith("mysql"):
        kwargs["pool_recycle"] = 3600

    return create_engine(settings.database_url, **kwargs)


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
