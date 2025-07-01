# app/database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config.settings import settings
from app.database.models import Base
import logging

logger = logging.getLogger(__name__)

# Sync engine for migrations
sync_engine = create_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql://"),
    echo=settings.DEBUG
)

# Async engine for application
async_engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG
)

AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=sync_engine)
    logger.info("Database tables created")

async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
