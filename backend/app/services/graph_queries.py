"""
Graph Query Service

Provides high-level graph query functions for simulation network analysis
using Neo4j and graph algorithms.
"""

from typing import Dict, List, Any, Optional
import networkx as nx

from app.db.neo4j import neo4j_client
from app.core.logging import logger


async def get_node_importance_scores(
    simulation_id: str,
    timestep: int,
    algorithm: str = "pagerank"
) -> List[Dict[str, Any]]:
    """
    Get node importance scores using various centrality algorithms

    Args:
        simulation_id: Simulation identifier
        timestep: Timestep to analyze
        algorithm: Algorithm to use ('pagerank', 'betweenness', 'eigenvector')

    Returns:
        List of nodes with importance scores
    """
    if algorithm == "pagerank":
        return await neo4j_client.run_pagerank_algorithm(
            simulation_id=simulation_id,
            timestep=timestep
        )

    elif algorithm == "betweenness":
        return await _compute_betweenness_centrality(
            simulation_id=simulation_id,
            timestep=timestep
        )

    elif algorithm == "eigenvector":
        return await _compute_eigenvector_centrality(
            simulation_id=simulation_id,
            timestep=timestep
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


async def find_critical_paths(
    simulation_id: str,
    timestep: int,
    source_node: Optional[str] = None,
    min_weight: float = 0.0,
    max_paths: int = 10
) -> List[Dict[str, Any]]:
    """
    Find critical paths in the network (high-weight contagion paths)

    Args:
        simulation_id: Simulation identifier
        timestep: Timestep to analyze
        source_node: Optional source node (if None, finds paths from all nodes)
        min_weight: Minimum path weight threshold
        max_paths: Maximum number of paths to return

    Returns:
        List of critical paths with metrics
    """
    async with neo4j_client.session() as session:
        if source_node:
            # Find paths from specific source
            query = """
            MATCH (source:SimNode {simulation_id: $simulation_id, timestep: $timestep, node_id: $source_node})
            MATCH path = (source)-[:EXPOSES*1..5]->(target:SimNode {simulation_id: $simulation_id, timestep: $timestep})
            WITH path,
                 reduce(w = 0, rel in relationships(path) | w + rel.weight) AS total_weight,
                 length(path) AS path_length
            WHERE total_weight >= $min_weight
            RETURN [node IN nodes(path) | node.node_id] AS path_nodes,
                   total_weight,
                   path_length
            ORDER BY total_weight DESC
            LIMIT $max_paths
            """
            result = await session.run(
                query,
                simulation_id=simulation_id,
                timestep=timestep,
                source_node=source_node,
                min_weight=min_weight,
                max_paths=max_paths
            )
        else:
            # Find critical paths across entire network
            query = """
            MATCH path = (source:SimNode {simulation_id: $simulation_id, timestep: $timestep})
                         -[:EXPOSES*1..4]->
                         (target:SimNode {simulation_id: $simulation_id, timestep: $timestep})
            WITH path,
                 reduce(w = 0, rel in relationships(path) | w + rel.weight) AS total_weight,
                 length(path) AS path_length,
                 source, target
            WHERE total_weight >= $min_weight AND source <> target
            RETURN [node IN nodes(path) | node.node_id] AS path_nodes,
                   source.node_id AS source_id,
                   target.node_id AS target_id,
                   total_weight,
                   path_length
            ORDER BY total_weight DESC
            LIMIT $max_paths
            """
            result = await session.run(
                query,
                simulation_id=simulation_id,
                timestep=timestep,
                min_weight=min_weight,
                max_paths=max_paths
            )

        records = await result.values()

    paths = []
    for record in records:
        path_data = {
            'path': record[0],
            'path_length': record[-1],
            'total_weight': float(record[-2])
        }

        if not source_node and len(record) > 3:
            path_data['source'] = record[1]
            path_data['target'] = record[2]

        paths.append(path_data)

    logger.info(
        "Found critical paths",
        simulation_id=simulation_id,
        timestep=timestep,
        path_count=len(paths)
    )

    return paths


async def detect_communities(
    simulation_id: str,
    timestep: int,
    algorithm: str = "louvain"
) -> Dict[str, Any]:
    """
    Detect communities in the network

    Args:
        simulation_id: Simulation identifier
        timestep: Timestep to analyze
        algorithm: Algorithm to use ('louvain', 'label_propagation')

    Returns:
        Community structure with statistics
    """
    if algorithm == "louvain":
        result = await neo4j_client.run_louvain_community_detection(
            simulation_id=simulation_id,
            timestep=timestep
        )
    elif algorithm == "label_propagation":
        result = await _run_label_propagation(
            simulation_id=simulation_id,
            timestep=timestep
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Calculate community statistics
    communities = result['communities']
    stats = {
        'num_communities': len(communities),
        'avg_community_size': sum(len(members) for members in communities.values()) / len(communities) if communities else 0,
        'largest_community_size': max(len(members) for members in communities.values()) if communities else 0,
        'smallest_community_size': min(len(members) for members in communities.values()) if communities else 0
    }

    logger.info(
        "Detected communities",
        simulation_id=simulation_id,
        timestep=timestep,
        num_communities=stats['num_communities']
    )

    return {
        'algorithm': algorithm,
        'statistics': stats,
        'communities': communities,
        'assignments': result['assignments']
    }


async def compute_clustering_coefficient(
    simulation_id: str,
    timestep: int,
    node_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute clustering coefficient for network or specific node

    Args:
        simulation_id: Simulation identifier
        timestep: Timestep to analyze
        node_id: Optional specific node (if None, computes global coefficient)

    Returns:
        Clustering coefficient(s)
    """
    # Load network from Neo4j
    graph_data = await neo4j_client.load_network_to_simulation(
        simulation_id=simulation_id,
        timestep=timestep
    )

    # Reconstruct NetworkX graph
    G = nx.DiGraph()
    for node in graph_data['nodes']:
        G.add_node(node['id'])

    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'])

    # Convert to undirected for clustering coefficient
    G_undirected = G.to_undirected()

    if node_id:
        # Node-specific clustering coefficient
        if node_id in G_undirected:
            coefficient = nx.clustering(G_undirected, node_id)
            return {
                'node_id': node_id,
                'clustering_coefficient': coefficient
            }
        else:
            raise ValueError(f"Node {node_id} not found in network")
    else:
        # Global clustering coefficient
        avg_clustering = nx.average_clustering(G_undirected)
        node_clustering = nx.clustering(G_undirected)

        # Get top nodes by clustering
        top_clustered = sorted(
            node_clustering.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'average_clustering_coefficient': avg_clustering,
            'top_clustered_nodes': [
                {'node_id': node, 'coefficient': coeff}
                for node, coeff in top_clustered
            ]
        }


async def compute_degree_distribution(
    simulation_id: str,
    timestep: int
) -> Dict[str, Any]:
    """
    Compute degree distribution statistics

    Args:
        simulation_id: Simulation identifier
        timestep: Timestep to analyze

    Returns:
        Degree distribution statistics
    """
    async with neo4j_client.session() as session:
        query = """
        MATCH (n:SimNode {simulation_id: $simulation_id, timestep: $timestep})
        OPTIONAL MATCH (n)-[r:EXPOSES {simulation_id: $simulation_id, timestep: $timestep}]->()
        WITH n, count(r) as out_degree
        OPTIONAL MATCH (n)<-[r2:EXPOSES {simulation_id: $simulation_id, timestep: $timestep}]-()
        WITH n, out_degree, count(r2) as in_degree
        RETURN n.node_id AS node_id,
               n.agent_type AS agent_type,
               out_degree,
               in_degree,
               out_degree + in_degree AS total_degree
        ORDER BY total_degree DESC
        """

        result = await session.run(
            query,
            simulation_id=simulation_id,
            timestep=timestep
        )
        records = await result.values()

    if not records:
        return {
            'node_count': 0,
            'avg_degree': 0,
            'max_degree': 0,
            'min_degree': 0
        }

    degrees = [record[4] for record in records]
    out_degrees = [record[2] for record in records]
    in_degrees = [record[3] for record in records]

    return {
        'node_count': len(records),
        'avg_degree': sum(degrees) / len(degrees),
        'avg_out_degree': sum(out_degrees) / len(out_degrees),
        'avg_in_degree': sum(in_degrees) / len(in_degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'top_nodes_by_degree': [
            {
                'node_id': record[0],
                'agent_type': record[1],
                'out_degree': record[2],
                'in_degree': record[3],
                'total_degree': record[4]
            }
            for record in records[:20]
        ]
    }


# Helper functions for additional algorithms

async def _compute_betweenness_centrality(
    simulation_id: str,
    timestep: int
) -> List[Dict[str, Any]]:
    """Compute betweenness centrality using Neo4j GDS"""
    graph_name = f"sim_{simulation_id}_t{timestep}_betweenness"

    async with neo4j_client.session() as session:
        # Create projection
        try:
            await session.run(
                """
                CALL gds.graph.project(
                    $graphName,
                    'SimNode',
                    'EXPOSES'
                )
                """,
                graphName=graph_name
            )
        except Exception as e:
            logger.debug(f"Graph projection exists: {e}")

        # Run betweenness
        query = """
        CALL gds.betweenness.stream($graphName)
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).agent_type AS agent_type,
               score
        ORDER BY score DESC
        """

        result = await session.run(query, graphName=graph_name)
        records = await result.values()

        # Clean up
        try:
            await session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
        except:
            pass

    return [
        {
            'node_id': record[0],
            'agent_type': record[1],
            'betweenness_score': float(record[2])
        }
        for record in records
    ]


async def _compute_eigenvector_centrality(
    simulation_id: str,
    timestep: int
) -> List[Dict[str, Any]]:
    """Compute eigenvector centrality using Neo4j GDS"""
    graph_name = f"sim_{simulation_id}_t{timestep}_eigenvector"

    async with neo4j_client.session() as session:
        # Create projection
        try:
            await session.run(
                """
                CALL gds.graph.project(
                    $graphName,
                    'SimNode',
                    'EXPOSES'
                )
                """,
                graphName=graph_name
            )
        except Exception as e:
            logger.debug(f"Graph projection exists: {e}")

        # Run eigenvector centrality
        query = """
        CALL gds.eigenvector.stream($graphName)
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).agent_type AS agent_type,
               score
        ORDER BY score DESC
        """

        result = await session.run(query, graphName=graph_name)
        records = await result.values()

        # Clean up
        try:
            await session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
        except:
            pass

    return [
        {
            'node_id': record[0],
            'agent_type': record[1],
            'eigenvector_score': float(record[2])
        }
        for record in records
    ]


async def _run_label_propagation(
    simulation_id: str,
    timestep: int
) -> Dict[str, Any]:
    """Run label propagation community detection"""
    graph_name = f"sim_{simulation_id}_t{timestep}_lpa"

    async with neo4j_client.session() as session:
        # Create projection
        try:
            await session.run(
                """
                CALL gds.graph.project(
                    $graphName,
                    'SimNode',
                    'EXPOSES',
                    {relationshipProperties: ['weight']}
                )
                """,
                graphName=graph_name
            )
        except Exception as e:
            logger.debug(f"Graph projection exists: {e}")

        # Run label propagation
        query = """
        CALL gds.labelPropagation.stream($graphName, {
            relationshipWeightProperty: 'weight'
        })
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).node_id AS node_id,
               gds.util.asNode(nodeId).agent_type AS agent_type,
               communityId
        """

        result = await session.run(query, graphName=graph_name)
        records = await result.values()

        # Clean up
        try:
            await session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
        except:
            pass

    communities = {}
    for record in records:
        node_id = record[0]
        community_id = record[2]

        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append({
            'node_id': node_id,
            'agent_type': record[1]
        })

    return {
        'num_communities': len(communities),
        'communities': communities,
        'assignments': [
            {'node_id': record[0], 'community_id': int(record[2])}
            for record in records
        ]
    }
