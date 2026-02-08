"""
Application configuration using Pydantic Settings
Loads from environment variables with fallback defaults
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "RUDRA"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Resilient Unified Decision & Risk Analytics"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 17170
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # PostgreSQL Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "rudra"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "rudra"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Neo4j Graph Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6380
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Simulation Settings
    MAX_SIMULATION_NODES: int = 10000
    MAX_SIMULATION_TIMESTEPS: int = 1000
    MONTE_CARLO_DEFAULT_RUNS: int = 1000
    SIMULATION_TIMEOUT_SECONDS: int = 3600
    
    # Logging
    LOG_LEVEL: str = "WARNING"  # Minimal logging - only warnings and errors
    LOG_FORMAT: str = "json"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()
