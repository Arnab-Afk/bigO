"""
Institution model - represents financial institutions in the network
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Enum, String, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.institution_state import InstitutionState
    from app.models.exposure import Exposure


class InstitutionType(str, enum.Enum):
    """Types of financial institutions"""
    BANK = "bank"
    CCP = "ccp"  # Central Counterparty Clearing House
    EXCHANGE = "exchange"
    CLEARING_HOUSE = "clearing_house"
    BROKER = "broker"
    ASSET_MANAGER = "asset_manager"
    INSURANCE = "insurance"
    HEDGE_FUND = "hedge_fund"
    SOVEREIGN = "sovereign"
    OTHER = "other"


class SystemicTier(str, enum.Enum):
    """Systemic importance classification"""
    G_SIB = "g_sib"      # Global Systemically Important Bank
    D_SIB = "d_sib"      # Domestic Systemically Important Bank
    TIER_1 = "tier_1"    # Large, interconnected
    TIER_2 = "tier_2"    # Medium importance
    TIER_3 = "tier_3"    # Smaller institutions


class Institution(Base):
    """
    Financial institution entity
    
    Represents banks, CCPs, exchanges, and other financial entities
    that participate in the network.
    """
    
    __tablename__ = "institutions"
    
    # Basic Information
    external_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="External identifier (e.g., LEI, BIC)"
    )
    
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Institution name"
    )
    
    short_name: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Short name or ticker"
    )
    
    # Classification
    type: Mapped[InstitutionType] = mapped_column(
        Enum(InstitutionType),
        nullable=False,
        default=InstitutionType.BANK,
        index=True,
        comment="Type of financial institution"
    )
    
    tier: Mapped[SystemicTier] = mapped_column(
        Enum(SystemicTier),
        nullable=False,
        default=SystemicTier.TIER_3,
        index=True,
        comment="Systemic importance tier"
    )
    
    # Geographic Information
    jurisdiction: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        index=True,
        comment="ISO 3166-1 alpha-2 country code"
    )
    
    region: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Geographic region"
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether institution is active in the network"
    )
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of the institution"
    )
    
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional metadata as JSON"
    )
    
    # Relationships
    states: Mapped[List["InstitutionState"]] = relationship(
        "InstitutionState",
        back_populates="institution",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    outbound_exposures: Mapped[List["Exposure"]] = relationship(
        "Exposure",
        foreign_keys="Exposure.source_institution_id",
        back_populates="source_institution",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    inbound_exposures: Mapped[List["Exposure"]] = relationship(
        "Exposure",
        foreign_keys="Exposure.target_institution_id",
        back_populates="target_institution",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self) -> str:
        return f"<Institution(id={self.id}, name='{self.name}', type={self.type.value})>"
