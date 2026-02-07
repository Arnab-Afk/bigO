"""
Institution State model - time-varying state of financial institutions
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Numeric, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.institution import Institution


class InstitutionState(Base):
    """
    Time-varying state of a financial institution
    
    Captures capital ratios, liquidity, exposure, and risk metrics
    at a specific point in time. Used for historical tracking and
    simulation state management.
    """
    
    __tablename__ = "institution_states"
    
    # Foreign Key
    institution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Timestamp for this state snapshot
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Point in time for this state"
    )
    
    # === Capital Metrics ===
    capital_ratio: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="Tier 1 capital ratio (capital / RWA)"
    )
    
    leverage_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="Leverage ratio (Tier 1 capital / total exposure)"
    )
    
    total_capital: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Total capital in base currency"
    )
    
    risk_weighted_assets: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Risk-weighted assets"
    )
    
    # === Liquidity Metrics ===
    liquidity_coverage_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="LCR: HQLA / 30-day net cash outflows"
    )
    
    net_stable_funding_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="NSFR: Available stable funding / Required stable funding"
    )
    
    liquidity_buffer: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        default=1.0,
        comment="Normalized liquidity buffer (0-1 scale)"
    )
    
    # === Exposure Metrics ===
    total_credit_exposure: Mapped[Decimal] = mapped_column(
        Numeric(20, 2),
        nullable=False,
        default=0,
        comment="Total credit exposure to counterparties"
    )
    
    total_market_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Total market risk exposure"
    )
    
    total_operational_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Total operational risk exposure"
    )
    
    # === Risk Metrics ===
    default_probability: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        default=0.01,
        comment="Estimated probability of default (0-1)"
    )
    
    stress_level: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        default=0.0,
        comment="Current stress level (0-1)"
    )
    
    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Composite risk score (0-100)"
    )
    
    # === Behavioral Parameters ===
    risk_appetite: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.5,
        comment="Risk appetite coefficient (0-1)"
    )
    
    margin_sensitivity: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.5,
        comment="Sensitivity to margin changes (0-1)"
    )
    
    # === Metadata ===
    source: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Data source (e.g., 'regulatory_filing', 'simulation')"
    )
    
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional metadata"
    )
    
    # Relationship
    institution: Mapped["Institution"] = relationship(
        "Institution",
        back_populates="states"
    )
    
    # Composite index for efficient time-series queries
    __table_args__ = (
        Index(
            "ix_institution_states_institution_timestamp",
            "institution_id",
            "timestamp",
            postgresql_using="btree"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<InstitutionState(institution_id={self.institution_id}, timestamp={self.timestamp}, stress={self.stress_level})>"
