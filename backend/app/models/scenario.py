"""
Scenario and Shock models - define simulation scenarios
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Enum, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.simulation import Simulation


class ShockType(str, enum.Enum):
    """Types of exogenous shocks"""
    INSTITUTION_DEFAULT = "institution_default"
    LIQUIDITY_FREEZE = "liquidity_freeze"
    MARKET_VOLATILITY = "market_volatility"
    MARGIN_CALL = "margin_call"
    CREDIT_DOWNGRADE = "credit_downgrade"
    OPERATIONAL_FAILURE = "operational_failure"
    REGULATORY_INTERVENTION = "regulatory_intervention"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    FX_SHOCK = "fx_shock"
    CYBER_ATTACK = "cyber_attack"


class Scenario(Base):
    """
    Simulation scenario definition
    
    Defines a set of conditions, shocks, and parameters for
    running simulations.
    """
    
    __tablename__ = "scenarios"
    
    # Basic Information
    name: Mapped[str] = mapped_column(
        nullable=False,
        index=True,
        comment="Scenario name"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Detailed description of the scenario"
    )
    
    # Classification
    category: Mapped[str] = mapped_column(
        nullable=False,
        default="custom",
        comment="Category (e.g., 'stress_test', 'historical', 'custom')"
    )
    
    is_template: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="Whether this is a template scenario"
    )
    
    # Simulation Parameters
    num_timesteps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        comment="Number of simulation timesteps"
    )
    
    convergence_threshold: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.000001,
        comment="Convergence threshold for early stopping"
    )
    
    # Risk Parameters
    base_volatility: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.15,
        comment="Base market volatility"
    )
    
    liquidity_premium: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.02,
        comment="Liquidity risk premium"
    )
    
    # Agent Behavior
    risk_aversion_min: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.3,
        comment="Minimum risk aversion coefficient"
    )
    
    risk_aversion_max: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.7,
        comment="Maximum risk aversion coefficient"
    )
    
    information_delay: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Information delay in timesteps"
    )
    
    # Shock Schedule
    shock_timing: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Mapping of timestep -> shock IDs"
    )
    
    # Additional Configuration
    parameters: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional scenario parameters"
    )
    
    # Relationships
    shocks: Mapped[List["Shock"]] = relationship(
        "Shock",
        back_populates="scenario",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    simulations: Mapped[List["Simulation"]] = relationship(
        "Simulation",
        back_populates="scenario",
        lazy="dynamic"
    )
    
    def __repr__(self) -> str:
        return f"<Scenario(id={self.id}, name='{self.name}')>"


class Shock(Base):
    """
    Exogenous shock definition
    
    Defines a specific shock event that can be applied
    during a simulation.
    """
    
    __tablename__ = "shocks"
    
    # Foreign Key
    scenario_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenarios.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Basic Information
    name: Mapped[str] = mapped_column(
        nullable=False,
        comment="Shock name"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Shock description"
    )
    
    # Shock Configuration
    shock_type: Mapped[ShockType] = mapped_column(
        Enum(ShockType),
        nullable=False,
        index=True,
        comment="Type of shock"
    )
    
    # Target
    target_type: Mapped[str] = mapped_column(
        nullable=False,
        default="institution",
        comment="Target type (institution, market, sector)"
    )
    
    target_id: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Target institution ID or sector name"
    )
    
    # Magnitude and Duration
    magnitude: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.5,
        comment="Shock magnitude (0-1 scale)"
    )
    
    duration: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Duration in timesteps"
    )
    
    # Timing
    trigger_timestep: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Timestep at which shock triggers"
    )
    
    # Additional Parameters
    parameters: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Shock-specific parameters"
    )
    
    # Relationship
    scenario: Mapped["Scenario"] = relationship(
        "Scenario",
        back_populates="shocks"
    )
    
    def __repr__(self) -> str:
        return f"<Shock(id={self.id}, type={self.shock_type.value}, magnitude={self.magnitude})>"
