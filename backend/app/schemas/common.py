"""
Common Pydantic schemas used across the application
"""

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# Generic type for paginated responses
T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints"""
    
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper"""
    
    items: List[T]
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    limit: int = Field(description="Items per page")
    pages: int = Field(description="Total number of pages")
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        limit: int
    ) -> "PaginatedResponse[T]":
        pages = (total + limit - 1) // limit if limit > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=pages
        )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Current server time")
    services: dict = Field(default_factory=dict, description="Status of dependent services")


class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    
    created_at: datetime
    updated_at: datetime


class IDMixin(BaseModel):
    """Mixin for ID field"""
    
    id: UUID
