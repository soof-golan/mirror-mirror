"""Database models and operations for Mirror Mirror using SQLModel."""

from datetime import datetime, UTC
from functools import partial
from typing import Optional

from sqlalchemy import func
from sqlmodel import Field, Session, SQLModel, create_engine, select

from mirror.config import config

# Models


class Prompt(SQLModel, table=True):
    """A prompt record from the database."""

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=UTC), index=True)
    source: str = Field(default="voice")


class Setting(SQLModel, table=True):
    """A key-value setting."""

    key: str = Field(primary_key=True)
    value: str


# Engine singleton
_engine = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{config.db_path}")
        SQLModel.metadata.create_all(_engine)
    return _engine


def init_db() -> None:
    """Initialize the database schema."""
    get_engine()


def get_latest_prompt() -> str | None:
    """Get the most recent prompt, or None if no prompts exist."""
    with Session(get_engine()) as session:
        statement = select(Prompt).order_by(Prompt.created_at.desc()).limit(1)
        result = session.exec(statement).first()
        return result.prompt if result else None


def get_current_prompt() -> str:
    """Get the current prompt (latest or default)."""
    return get_latest_prompt() or config.default_prompt


def get_random_prompt() -> str:
    """Get a random prompt from the database, or default if none exist."""
    with Session(get_engine()) as session:
        statement = select(Prompt).order_by(func.random()).limit(1)
        result = session.exec(statement).first()
        return result.prompt if result else config.default_prompt


def save_prompt(prompt: str, source: str = "voice") -> int:
    """Save a new prompt and return its ID."""
    with Session(get_engine()) as session:
        db_prompt = Prompt(prompt=prompt, source=source)
        session.add(db_prompt)
        session.commit()
        session.refresh(db_prompt)
        return db_prompt.id


def get_prompt_history(limit: int = 10) -> list[Prompt]:
    """Get recent prompt history."""
    with Session(get_engine()) as session:
        statement = select(Prompt).order_by(Prompt.created_at.desc()).limit(limit)
        return list(session.exec(statement).all())


def get_setting(key: str, default: str | None = None) -> str | None:
    """Get a setting value."""
    with Session(get_engine()) as session:
        result = session.get(Setting, key)
        return result.value if result else default


def set_setting(key: str, value: str) -> None:
    """Set a setting value."""
    with Session(get_engine()) as session:
        existing = session.get(Setting, key)
        if existing:
            existing.value = value
        else:
            session.add(Setting(key=key, value=value))
        session.commit()
