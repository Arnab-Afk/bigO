"""
Simulation model - represents simulation runs and results
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
    from app.models.scenario import Scenario


class SimulationStatus(str, enum.Enum):
    """Status of a simulation run"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Simulation(Base):
    """
    Simulation run record
    
    Tracks the execution of a simulation scenario, including
    its status, progress, and results.
    """
    
    __tablename__ = "simulations"
    
    # Scenario Reference
    scenario_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenarios.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Reference to the scenario definition"
    )
    
    # Basic Information
    name: Mapped[str] = mapped_column(
        nullable=False,
        comment="Name of this simulation run"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of the simulation"
    )
    
    # Status
    status: Mapped[SimulationStatus] = mapped_column(
        Enum(SimulationStatus),
        nullable=False,
        default=SimulationStatus.PENDING,
        index=True,
        comment="Current status of the simulation"
    )
    
    # Progress Tracking
    current_timestep: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Current timestep in the simulation"
    )
    
    total_timesteps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        comment="Total planned timesteps"
    )
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="When simulation execution started"
    )
    
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="When simulation execution completed"
    )
    
    # Configuration
    parameters: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Simulation parameters"
    )
    
    # Error Information
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if simulation failed"
    )
    
    # Relationships
    scenario: Mapped[Optional["Scenario"]] = relationship(
        "Scenario",
        back_populates="simulations"
    )
    
    results: Mapped[List["SimulationResult"]] = relationship(
        "SimulationResult",
        back_populates="simulation",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    @property
    def progress(self) -> float:
        """Calculate simulation progress as percentage"""
        if self.total_timesteps == 0:
            return 0.0
        return (self.current_timestep / self.total_timesteps) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate simulation duration in seconds"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def __repr__(self) -> str:
        return f"<Simulation(id={self.id}, name='{self.name}', status={self.status.value})>"


class SimulationResult(Base):
    """
    Simulation results and metrics
    
    Stores the output of a completed simulation, including
    aggregate metrics and detailed timeline data.
    """
    
    __tablename__ = "simulation_results"
    
    # Foreign Key
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Result Type
    result_type: Mapped[str] = mapped_column(
        nullable=False,
        default="final_metrics",
        comment="Type of result (final_metrics, timeline, cascade_analysis)"
    )
    
    # Metrics
    total_defaults: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of institution defaults"
    )
    
    max_cascade_depth: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Maximum cascade depth observed"
    )
    
    survival_rate: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=1.0,
        comment="Proportion of institutions that survived"
    )
    
    final_systemic_stress: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.0,
        comment="Final aggregate systemic stress level"
    )
    
    total_system_loss: Mapped[Decimal] = mapped_column(
        nullable=False,
        default=0.0,
        comment="Total system-wide losses"
    )
    
    time_to_first_default: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Timestep of first default (-1 if none)"
    )
    
    # Detailed Data
    metrics_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed metrics as JSON"
    )
    
    timeline_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Timestep-by-timestep data"
    )
    
    cascade_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Cascade analysis data"
    )
    
    # Relationship
    simulation: Mapped["Simulation"] = relationship(
        "Simulation",
        back_populates="results"
    )
    
    def __repr__(self) -> str:
        return f"<SimulationResult(simulation_id={self.simulation_id}, defaults={self.total_defaults})>"
