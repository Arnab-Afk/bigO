"""
Exposure model - represents financial exposures between institutions
"""

import enum
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum, ForeignKey, Numeric, Date, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.institution import Institution


class ExposureType(str, enum.Enum):
    """Types of financial exposures"""
    INTERBANK_LENDING = "interbank_lending"
    DERIVATIVES = "derivatives"
    REPO = "repo"
    SECURITIES_LENDING = "securities_lending"
    SETTLEMENT = "settlement"
    CLEARING_MARGIN = "clearing_margin"
    CREDIT_LINE = "credit_line"
    TRADE_FINANCE = "trade_finance"
    COLLATERAL = "collateral"
    OTHER = "other"


class Exposure(Base):
    """
    Financial exposure between two institutions
    
    Represents a directed edge in the financial network, capturing
    the credit, settlement, or other financial relationship between
    a source and target institution.
    """
    
    __tablename__ = "exposures"
    
    # Foreign Keys
    source_institution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Institution with the exposure (creditor)"
    )
    
    target_institution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Institution being exposed to (debtor)"
    )
    
    # Exposure Classification
    exposure_type: Mapped[ExposureType] = mapped_column(
        Enum(ExposureType),
        nullable=False,
        default=ExposureType.INTERBANK_LENDING,
        index=True,
        comment="Type of financial exposure"
    )
    
    # === Exposure Amounts ===
    gross_exposure: Mapped[Decimal] = mapped_column(
        Numeric(20, 2),
        nullable=False,
        comment="Gross exposure amount in base currency"
    )
    
    net_exposure: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Net exposure after netting agreements"
    )
    
    # === Collateral & Risk Mitigation ===
    collateral_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        default=0,
        comment="Value of collateral posted"
    )
    
    collateral_haircut: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.0,
        comment="Haircut applied to collateral (0-1)"
    )
    
    # === Risk Parameters ===
    recovery_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.55,
        comment="Expected recovery rate in case of default (0-1)"
    )
    
    contagion_probability: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.1,
        comment="Probability of stress transmission (0-1)"
    )
    
    settlement_urgency: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0.5,
        comment="Settlement time criticality (0-1)"
    )
    
    # === Temporal Information ===
    effective_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment="Date exposure became effective"
    )
    
    maturity_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        comment="Maturity date of the exposure"
    )
    
    # === Validity Period (for temporal queries) ===
    valid_from: Mapped[datetime] = mapped_column(
        nullable=False,
        comment="Start of validity period"
    )
    
    valid_to: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="End of validity period (null = current)"
    )
    
    # === Reference Data ===
    netting_agreement_id: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Reference to netting agreement"
    )
    
    currency: Mapped[str] = mapped_column(
        nullable=False,
        default="USD",
        comment="Currency of exposure"
    )
    
    # === Metadata ===
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
        comment="Additional exposure metadata"
    )
    
    # Relationships
    source_institution: Mapped["Institution"] = relationship(
        "Institution",
        foreign_keys=[source_institution_id],
        back_populates="outbound_exposures"
    )
    
    target_institution: Mapped["Institution"] = relationship(
        "Institution",
        foreign_keys=[target_institution_id],
        back_populates="inbound_exposures"
    )
    
    # Constraints and Indexes
    __table_args__ = (
        # Ensure source and target are different
        CheckConstraint(
            "source_institution_id != target_institution_id",
            name="ck_exposure_different_institutions"
        ),
        # Composite index for edge lookups
        Index(
            "ix_exposures_source_target",
            "source_institution_id",
            "target_institution_id",
            postgresql_using="btree"
        ),
        # Index for valid exposures
        Index(
            "ix_exposures_valid",
            "valid_from",
            "valid_to",
            postgresql_using="btree"
        ),
    )
    
    @property
    def effective_exposure(self) -> Decimal:
        """Calculate exposure after collateral"""
        collateral_effective = self.collateral_value or Decimal(0)
        collateral_after_haircut = collateral_effective * (1 - self.collateral_haircut)
        net = self.net_exposure or self.gross_exposure
        return max(Decimal(0), net - collateral_after_haircut)
    
    @property
    def loss_given_default(self) -> Decimal:
        """Calculate expected loss given default"""
        return self.effective_exposure * (1 - self.recovery_rate)
    
    def __repr__(self) -> str:
        return f"<Exposure(source={self.source_institution_id}, target={self.target_institution_id}, type={self.exposure_type.value}, amount={self.gross_exposure})>"
