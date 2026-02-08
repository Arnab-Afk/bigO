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
from app.engine.network import NetworkAnalyzer, build_network_graph
from app.engine.contagion import ContagionPropagator

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


@router.post("/analyze")
async def compute_advanced_network_analysis(
    min_exposure: Optional[float] = Query(None, description="Minimum exposure filter"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Compute advanced network metrics using graph algorithms.
    
    Includes betweenness, eigenvector, PageRank, and systemic risk indicators.
    """
    from datetime import datetime
    
    # Load network data
    inst_query = select(Institution).where(Institution.is_active == True)
    inst_result = await db.execute(inst_query)
    institutions = inst_result.scalars().all()
    
    now = datetime.utcnow()
    exp_query = select(Exposure).where(
        or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
    )
    if min_exposure:
        exp_query = exp_query.where(Exposure.gross_exposure >= min_exposure)
    
    exp_result = await db.execute(exp_query)
    exposures = exp_result.scalars().all()
    
    # Build network graph
    institutions_data = [
        {
            'id': str(inst.id),
            'name': inst.name,
            'type': inst.type.value,
            'tier': inst.tier.value,
        }
        for inst in institutions
    ]
    
    exposures_data = [
        {
            'source_institution_id': str(exp.source_institution_id),
            'target_institution_id': str(exp.target_institution_id),
            'exposure_type': exp.exposure_type.value,
            'gross_exposure': float(exp.gross_exposure),
            'contagion_probability': float(exp.contagion_probability),
            'recovery_rate': float(exp.recovery_rate),
            'settlement_urgency': float(exp.settlement_urgency),
        }
        for exp in exposures
    ]
    
    network = build_network_graph(institutions_data, exposures_data)
    
    # Initialize analyzer
    analyzer = NetworkAnalyzer(network)
    
    # Compute all centralities
    centralities = analyzer.compute_all_centralities()
    
    # Compute network-level metrics
    network_metrics = analyzer.compute_network_metrics()
    
    # Find bottlenecks
    bottlenecks = analyzer.identify_bottlenecks(top_k=10)
    
    return {
        "network_metrics": {
            "node_count": network_metrics.node_count,
            "edge_count": network_metrics.edge_count,
            "density": round(network_metrics.density, 4),
            "average_clustering": round(network_metrics.average_clustering, 4),
            "average_path_length": round(network_metrics.average_path_length, 4) if network_metrics.average_path_length else None,
            "diameter": network_metrics.diameter,
            "concentration_index": round(network_metrics.concentration_index, 4),
            "interconnectedness_score": round(network_metrics.interconnectedness_score, 4),
            "complexity_score": round(network_metrics.complexity_score, 4),
        },
        "top_centralities": [
            {
                "institution_id": str(node_id),
                "degree_centrality": round(cent.degree_centrality, 4),
                "betweenness_centrality": round(cent.betweenness_centrality, 4),
                "eigenvector_centrality": round(cent.eigenvector_centrality, 4),
                "pagerank": round(cent.pagerank, 4),
                "systemic_importance": round(cent.systemic_importance, 4),
            }
            for node_id, cent in sorted(
                centralities.items(),
                key=lambda x: x[1].systemic_importance,
                reverse=True
            )[:20]
        ],
        "bottleneck_nodes": [
            {
                "institution_id": str(node_id),
                "impact_score": round(score, 4),
            }
            for node_id, score in bottlenecks
        ],
    }


@router.post("/contagion-paths")
async def find_contagion_paths(
    source_id: UUID,
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    max_length: int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Find critical contagion paths from a source institution.
    """
    from datetime import datetime
    
    # Verify source institution exists
    inst_result = await db.execute(
        select(Institution).where(Institution.id == source_id)
    )
    source_inst = inst_result.scalar_one_or_none()
    
    if not source_inst:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {source_id} not found"
        )
    
    # Load network data
    inst_query = select(Institution).where(Institution.is_active == True)
    institutions = (await db.execute(inst_query)).scalars().all()
    
    now = datetime.utcnow()
    exp_query = select(Exposure).where(
        or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
    )
    exposures = (await db.execute(exp_query)).scalars().all()
    
    # Build network
    institutions_data = [
        {'id': str(i.id), 'name': i.name, 'type': i.type.value, 'tier': i.tier.value}
        for i in institutions
    ]
    exposures_data = [
        {
            'source_institution_id': str(e.source_institution_id),
            'target_institution_id': str(e.target_institution_id),
            'exposure_type': e.exposure_type.value,
            'gross_exposure': float(e.gross_exposure),
            'contagion_probability': float(e.contagion_probability),
            'recovery_rate': float(e.recovery_rate),
            'settlement_urgency': float(e.settlement_urgency),
        }
        for e in exposures
    ]
    
    network = build_network_graph(institutions_data, exposures_data)
    analyzer = NetworkAnalyzer(network)
    
    # Find critical paths
    paths = analyzer.find_critical_paths(
        source=source_id,
        threshold=threshold,
        max_length=max_length
    )
    
    return {
        "source_institution_id": str(source_id),
        "source_name": source_inst.name,
        "threshold": threshold,
        "paths_found": len(paths),
        "critical_paths": [
            {
                "path": [str(node_id) for node_id in path.path],
                "probability": round(path.probability, 4),
                "total_exposure": float(path.total_exposure),
                "path_length": path.path_length,
                "risk_score": round(path.risk_score, 4),
            }
            for path in paths[:50]  # Limit to top 50
        ],
    }


@router.post("/cascade-simulation")
async def simulate_cascade(
    shocked_institutions: List[UUID],
    max_rounds: int = Query(10, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Simulate contagion cascade from initial shock.
    """
    from datetime import datetime

    # Load network
    inst_query = select(Institution).where(Institution.is_active == True)
    institutions = (await db.execute(inst_query)).scalars().all()

    now = datetime.utcnow()
    exp_query = select(Exposure).where(
        or_(Exposure.valid_to.is_(None), Exposure.valid_to > now)
    )
    exposures = (await db.execute(exp_query)).scalars().all()

    # Build network
    institutions_data = [
        {'id': str(i.id), 'name': i.name, 'type': i.type.value, 'tier': i.tier.value}
        for i in institutions
    ]
    exposures_data = [
        {
            'source_institution_id': str(e.source_institution_id),
            'target_institution_id': str(e.target_institution_id),
            'exposure_type': e.exposure_type.value,
            'gross_exposure': float(e.gross_exposure),
            'contagion_probability': float(e.contagion_probability),
            'recovery_rate': float(e.recovery_rate),
            'settlement_urgency': float(e.settlement_urgency),
        }
        for e in exposures
    ]

    network = build_network_graph(institutions_data, exposures_data)
    propagator = ContagionPropagator(network)

    # Create initial state
    from app.engine.contagion import PropagationState

    initial_state = PropagationState(
        capital_levels={UUID(i['id']): 10000.0 for i in institutions_data},
        liquidity_levels={UUID(i['id']): 0.5 for i in institutions_data},
        stress_levels={UUID(i['id']): 0.1 for i in institutions_data},
        defaulted=set(),
    )

    # Run cascade
    final_state, cascade_history = propagator.propagate_shock(
        initial_state=initial_state,
        shocked_institutions=shocked_institutions,
        max_rounds=max_rounds
    )

    return {
        "initial_shocks": [str(inst_id) for inst_id in shocked_institutions],
        "cascade_rounds": len(cascade_history),
        "total_defaults": len(final_state.defaulted),
        "total_losses": float(sum(c.total_losses for c in cascade_history)),
        "cascade_history": [
            {
                "round": cascade.round_number,
                "defaults": len(cascade.defaults),
                "losses": float(cascade.total_losses),
                "affected": [str(inst_id) for inst_id in cascade.affected_institutions],
            }
            for cascade in cascade_history
        ],
        "final_defaulted": [str(inst_id) for inst_id in final_state.defaulted],
    }


# ========== Neo4j-powered Graph Algorithm Endpoints ==========


@router.get("/{sim_id}/communities")
async def get_simulation_communities(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    algorithm: str = Query("louvain", description="Algorithm: louvain or label_propagation"),
) -> dict:
    """
    Detect communities in simulation network using Neo4j GDS

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        algorithm: Community detection algorithm

    Returns:
        Community structure and statistics
    """
    from app.services.graph_queries import detect_communities

    try:
        result = await detect_communities(
            simulation_id=sim_id,
            timestep=timestep,
            algorithm=algorithm
        )

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "algorithm": result['algorithm'],
            "statistics": result['statistics'],
            "communities": result['communities'],
            "assignments": result['assignments']
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect communities: {str(e)}"
        )


@router.get("/{sim_id}/paths/{source}/{target}")
async def get_shortest_path(
    sim_id: str,
    source: str,
    target: str,
    timestep: int = Query(..., description="Simulation timestep"),
) -> dict:
    """
    Find shortest path between two nodes in simulation network

    Args:
        sim_id: Simulation identifier
        source: Source node ID
        target: Target node ID
        timestep: Timestep to analyze

    Returns:
        Shortest path information
    """
    from app.db.neo4j import neo4j_client

    try:
        path_result = await neo4j_client.run_shortest_path_analysis(
            simulation_id=sim_id,
            timestep=timestep,
            source_node=source,
            target_node=target
        )

        if not path_result:
            return {
                "simulation_id": sim_id,
                "timestep": timestep,
                "source": source,
                "target": target,
                "path_found": False,
                "message": "No path exists between the specified nodes"
            }

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "source": source,
            "target": target,
            "path_found": True,
            "path": path_result['path'],
            "path_length": path_result['length'],
            "total_weight": path_result['total_weight']
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find path: {str(e)}"
        )


@router.get("/{sim_id}/gds/pagerank")
async def run_pagerank(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    damping_factor: float = Query(0.85, ge=0.0, le=1.0, description="PageRank damping factor"),
    max_iterations: int = Query(20, ge=1, le=100, description="Maximum iterations"),
    top_k: int = Query(50, ge=1, le=500, description="Number of top nodes to return"),
) -> dict:
    """
    Run PageRank algorithm on simulation network using Neo4j GDS

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        damping_factor: PageRank damping factor (default 0.85)
        max_iterations: Maximum iterations
        top_k: Number of top nodes to return

    Returns:
        PageRank scores for nodes
    """
    from app.db.neo4j import neo4j_client

    try:
        result = await neo4j_client.run_pagerank_algorithm(
            simulation_id=sim_id,
            timestep=timestep,
            dampingFactor=damping_factor,
            max_iterations=max_iterations
        )

        # Return top K nodes
        top_nodes = result[:top_k]

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "algorithm": "pagerank",
            "parameters": {
                "damping_factor": damping_factor,
                "max_iterations": max_iterations
            },
            "node_count": len(result),
            "top_nodes": top_nodes
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run PageRank: {str(e)}"
        )


@router.get("/{sim_id}/gds/betweenness")
async def run_betweenness_centrality(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    top_k: int = Query(50, ge=1, le=500, description="Number of top nodes to return"),
) -> dict:
    """
    Run betweenness centrality on simulation network using Neo4j GDS

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        top_k: Number of top nodes to return

    Returns:
        Betweenness centrality scores
    """
    from app.services.graph_queries import get_node_importance_scores

    try:
        result = await get_node_importance_scores(
            simulation_id=sim_id,
            timestep=timestep,
            algorithm="betweenness"
        )

        top_nodes = result[:top_k]

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "algorithm": "betweenness_centrality",
            "node_count": len(result),
            "top_nodes": top_nodes
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute betweenness centrality: {str(e)}"
        )


@router.get("/{sim_id}/gds/eigenvector")
async def run_eigenvector_centrality(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    top_k: int = Query(50, ge=1, le=500, description="Number of top nodes to return"),
) -> dict:
    """
    Run eigenvector centrality on simulation network using Neo4j GDS

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        top_k: Number of top nodes to return

    Returns:
        Eigenvector centrality scores
    """
    from app.services.graph_queries import get_node_importance_scores

    try:
        result = await get_node_importance_scores(
            simulation_id=sim_id,
            timestep=timestep,
            algorithm="eigenvector"
        )

        top_nodes = result[:top_k]

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "algorithm": "eigenvector_centrality",
            "node_count": len(result),
            "top_nodes": top_nodes
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute eigenvector centrality: {str(e)}"
        )


@router.get("/{sim_id}/gds/critical-paths")
async def get_critical_paths(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    source_node: Optional[str] = Query(None, description="Optional source node"),
    min_weight: float = Query(0.0, ge=0.0, description="Minimum path weight"),
    max_paths: int = Query(10, ge=1, le=100, description="Maximum paths to return"),
) -> dict:
    """
    Find critical contagion paths in simulation network

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        source_node: Optional source node ID
        min_weight: Minimum path weight threshold
        max_paths: Maximum number of paths to return

    Returns:
        Critical paths with metrics
    """
    from app.services.graph_queries import find_critical_paths

    try:
        paths = await find_critical_paths(
            simulation_id=sim_id,
            timestep=timestep,
            source_node=source_node,
            min_weight=min_weight,
            max_paths=max_paths
        )

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "source_node": source_node,
            "parameters": {
                "min_weight": min_weight,
                "max_paths": max_paths
            },
            "paths_found": len(paths),
            "critical_paths": paths
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find critical paths: {str(e)}"
        )


@router.get("/{sim_id}/clustering")
async def get_clustering_coefficient(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
    node_id: Optional[str] = Query(None, description="Optional specific node"),
) -> dict:
    """
    Compute clustering coefficient for network or specific node

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze
        node_id: Optional specific node ID

    Returns:
        Clustering coefficient(s)
    """
    from app.services.graph_queries import compute_clustering_coefficient

    try:
        result = await compute_clustering_coefficient(
            simulation_id=sim_id,
            timestep=timestep,
            node_id=node_id
        )

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            "node_id": node_id,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute clustering coefficient: {str(e)}"
        )


@router.get("/{sim_id}/degree-distribution")
async def get_degree_distribution(
    sim_id: str,
    timestep: int = Query(..., description="Simulation timestep"),
) -> dict:
    """
    Get degree distribution statistics for simulation network

    Args:
        sim_id: Simulation identifier
        timestep: Timestep to analyze

    Returns:
        Degree distribution statistics
    """
    from app.services.graph_queries import compute_degree_distribution

    try:
        result = await compute_degree_distribution(
            simulation_id=sim_id,
            timestep=timestep
        )

        return {
            "simulation_id": sim_id,
            "timestep": timestep,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute degree distribution: {str(e)}"
        )
