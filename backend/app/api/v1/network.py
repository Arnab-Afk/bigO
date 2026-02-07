"""
Network analysis API endpoints
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.institution import Institution
from app.models.exposure import Exposure
from app.models.institution_state import InstitutionState

router = APIRouter()


@router.get("/graph")
async def get_network_graph(
    format: str = Query("adjacency", description="Output format"),
    include_weights: bool = Query(True, description="Include edge weights"),
    min_exposure: Optional[float] = Query(None, description="Minimum exposure filter"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get the financial network graph representation.
    """
    from datetime import datetime
    
    # Get all active institutions
    inst_query = select(Institution).where(Institution.is_active == True)
    inst_result = await db.execute(inst_query)
    institutions = inst_result.scalars().all()
    
    # Get all active exposures
    now = datetime.utcnow()
    exp_query = select(Exposure).where(
        or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
    )
    if min_exposure:
        exp_query = exp_query.where(Exposure.gross_exposure >= min_exposure)
    
    exp_result = await db.execute(exp_query)
    exposures = exp_result.scalars().all()
    
    # Build nodes
    nodes = []
    for inst in institutions:
        node_data = {
            "id": str(inst.id),
            "external_id": inst.external_id,
            "name": inst.name,
            "type": inst.type.value,
            "tier": inst.tier.value,
            "jurisdiction": inst.jurisdiction,
        }
        nodes.append(node_data)
    
    # Build edges
    edges = []
    for exp in exposures:
        edge_data = {
            "source": str(exp.source_institution_id),
            "target": str(exp.target_institution_id),
            "exposure_type": exp.exposure_type.value,
        }
        if include_weights:
            edge_data.update({
                "gross_exposure": float(exp.gross_exposure),
                "contagion_probability": float(exp.contagion_probability),
                "recovery_rate": float(exp.recovery_rate),
                "settlement_urgency": float(exp.settlement_urgency),
            })
        edges.append(edge_data)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "format": format,
        }
    }


@router.get("/metrics")
async def get_network_metrics(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get network-level metrics and statistics.
    """
    from datetime import datetime
    
    # Count nodes
    node_count = await db.scalar(
        select(func.count(Institution.id)).where(Institution.is_active == True)
    )
    
    # Count edges
    now = datetime.utcnow()
    edge_count = await db.scalar(
        select(func.count(Exposure.id)).where(
            or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
        )
    )
    
    # Calculate density
    max_edges = node_count * (node_count - 1) if node_count > 1 else 1
    density = edge_count / max_edges if max_edges > 0 else 0
    
    # Get total exposure
    total_exposure = await db.scalar(
        select(func.sum(Exposure.gross_exposure)).where(
            or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
        )
    ) or 0
    
    # Get exposure concentration (HHI)
    exposure_by_institution = await db.execute(
        select(
            Exposure.source_institution_id,
            func.sum(Exposure.gross_exposure).label("total")
        )
        .where(or_(Exposure.valid_to.is_(None), Exposure.valid_to > now))
        .group_by(Exposure.source_institution_id)
    )
    exposures = exposure_by_institution.all()
    
    if exposures and total_exposure > 0:
        hhi = sum((float(e.total) / float(total_exposure)) ** 2 for e in exposures)
    else:
        hhi = 0
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(density, 4),
        "total_exposure": float(total_exposure),
        "hhi_concentration": round(hhi, 4),
        "avg_degree": round(edge_count / node_count, 2) if node_count > 0 else 0,
    }


@router.get("/centrality/{institution_id}")
async def get_institution_centrality(
    institution_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get centrality metrics for a specific institution.
    
    Note: For full centrality computation, use the simulation engine.
    This provides basic degree-based metrics.
    """
    from datetime import datetime
    
    # Verify institution exists
    inst_query = select(Institution).where(Institution.id == institution_id)
    inst_result = await db.execute(inst_query)
    institution = inst_result.scalar_one_or_none()
    
    if not institution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {institution_id} not found"
        )
    
    now = datetime.utcnow()
    valid_filter = or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
    
    # Count outbound edges (out-degree)
    out_degree = await db.scalar(
        select(func.count(Exposure.id))
        .where(Exposure.source_institution_id == institution_id)
        .where(valid_filter)
    )
    
    # Count inbound edges (in-degree)
    in_degree = await db.scalar(
        select(func.count(Exposure.id))
        .where(Exposure.target_institution_id == institution_id)
        .where(valid_filter)
    )
    
    # Sum outbound exposure
    out_exposure = await db.scalar(
        select(func.sum(Exposure.gross_exposure))
        .where(Exposure.source_institution_id == institution_id)
        .where(valid_filter)
    ) or 0
    
    # Sum inbound exposure
    in_exposure = await db.scalar(
        select(func.sum(Exposure.gross_exposure))
        .where(Exposure.target_institution_id == institution_id)
        .where(valid_filter)
    ) or 0
    
    # Get total node count for normalization
    total_nodes = await db.scalar(
        select(func.count(Institution.id)).where(Institution.is_active == True)
    )
    
    # Calculate normalized degree centrality
    degree_centrality = (in_degree + out_degree) / (2 * (total_nodes - 1)) if total_nodes > 1 else 0
    
    return {
        "institution_id": str(institution_id),
        "institution_name": institution.name,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": out_degree + in_degree,
        "degree_centrality": round(degree_centrality, 4),
        "outbound_exposure": float(out_exposure),
        "inbound_exposure": float(in_exposure),
        "net_exposure": float(out_exposure - in_exposure),
        "note": "For advanced centrality metrics (betweenness, eigenvector), run simulation analysis"
    }


@router.get("/systemic-importance")
async def get_systemic_importance_ranking(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get ranking of institutions by systemic importance.
    
    Uses a simplified score based on tier, exposure, and connectivity.
    """
    from datetime import datetime
    
    now = datetime.utcnow()
    
    # Get all active institutions with their exposure totals
    query = """
    SELECT 
        i.id,
        i.name,
        i.type,
        i.tier,
        COALESCE(out_exp.total_out, 0) as outbound_exposure,
        COALESCE(in_exp.total_in, 0) as inbound_exposure,
        COALESCE(out_exp.out_count, 0) as out_degree,
        COALESCE(in_exp.in_count, 0) as in_degree
    FROM institutions i
    LEFT JOIN (
        SELECT source_institution_id, 
               SUM(gross_exposure) as total_out,
               COUNT(*) as out_count
        FROM exposures
        WHERE valid_to IS NULL OR valid_to > :now
        GROUP BY source_institution_id
    ) out_exp ON i.id = out_exp.source_institution_id
    LEFT JOIN (
        SELECT target_institution_id,
               SUM(gross_exposure) as total_in,
               COUNT(*) as in_count
        FROM exposures
        WHERE valid_to IS NULL OR valid_to > :now
        GROUP BY target_institution_id
    ) in_exp ON i.id = in_exp.target_institution_id
    WHERE i.is_active = true
    """
    
    # For simplicity, use ORM queries
    institutions = await db.execute(
        select(Institution).where(Institution.is_active == True)
    )
    institutions = institutions.scalars().all()
    
    # Calculate simple systemic importance score
    rankings = []
    for inst in institutions:
        # Get exposure data
        out_exp = await db.scalar(
            select(func.sum(Exposure.gross_exposure))
            .where(Exposure.source_institution_id == inst.id)
            .where(or_(Exposure.valid_to.is_(None), Exposure.valid_to > now))
        ) or 0
        
        in_exp = await db.scalar(
            select(func.sum(Exposure.gross_exposure))
            .where(Exposure.target_institution_id == inst.id)
            .where(or_(Exposure.valid_to.is_(None), Exposure.valid_to > now))
        ) or 0
        
        # Tier weight
        tier_weights = {
            "g_sib": 1.0,
            "d_sib": 0.8,
            "tier_1": 0.6,
            "tier_2": 0.4,
            "tier_3": 0.2,
        }
        tier_weight = tier_weights.get(inst.tier.value, 0.2)
        
        # Simple score (can be refined)
        total_exposure = float(out_exp + in_exp)
        score = tier_weight * 50 + (total_exposure / 1e9) * 50  # Normalize to billions
        
        rankings.append({
            "institution_id": str(inst.id),
            "name": inst.name,
            "type": inst.type.value,
            "tier": inst.tier.value,
            "total_exposure": total_exposure,
            "systemic_importance_score": round(min(score, 100), 2),
        })
    
    # Sort by score
    rankings.sort(key=lambda x: x["systemic_importance_score"], reverse=True)
    
    return {
        "rankings": rankings[:limit],
        "total_institutions": len(rankings),
    }
