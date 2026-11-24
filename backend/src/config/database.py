from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from src.config.settings import DB_URL

class Base(DeclarativeBase):
    pass

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=240)
Session = sessionmaker(engine, expire_on_commit=False)


def patch_schema():
    """Add columns that create_all cannot add to an existing table."""
    if engine.dialect.name != 'postgresql':
        return
    with engine.begin() as connection:
        connection.execute(text('ALTER TABLE runs ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES runs(id)'))


def connection():
    with Session() as db:
        yield db
