"""
Neo4j Graph Database Integration

Provides graph database operations for complex network queries and analysis.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.core.config import settings
from app.core.logging import logger


class Neo4jClient:
    """
    Neo4j graph database client for network analysis
    
    Handles connections and queries to Neo4j for complex graph operations
    that benefit from native graph database performance.
    """
    
    def __init__(self):
        """Initialize Neo4j client"""
        self.driver: Optional[AsyncDriver] = None
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
    
    async def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j", uri=self.uri)
        except (ServiceUnavailable, AuthError) as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Closed Neo4j connection")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for Neo4j session"""
        if not self.driver:
            await self.connect()
        
        async with self.driver.session() as session:
            yield session
    
    async def create_institution_node(
        self,
        institution_id: UUID,
        properties: Dict[str, Any]
    ):
        """
        Create an institution node
        
        Args:
            institution_id: Institution UUID
            properties: Node properties (name, type, tier, etc.)
        """
        query = """
        MERGE (i:Institution {id: $id})
        SET i += $properties
        RETURN i
        """
        
        async with self.session() as session:
            result = await session.run(
                query,
                id=str(institution_id),
                properties=properties
            )
            await result.consume()
    
    async def create_exposure_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        exposure_type: str,
        properties: Dict[str, Any]
    ):
        """
        Create exposure relationship between institutions
        
        Args:
            source_id: Source institution UUID
            target_id: Target institution UUID
            exposure_type: Type of exposure
            properties: Edge properties (amount, probability, etc.)
        """
        query = """
        MATCH (source:Institution {id: $source_id})
        MATCH (target:Institution {id: $target_id})
        MERGE (source)-[e:HAS_EXPOSURE {type: $exposure_type}]->(target)
        SET e += $properties
        RETURN e
        """
        
        async with self.session() as session:
            result = await session.run(
                query,
                source_id=str(source_id),
                target_id=str(target_id),
                exposure_type=exposure_type,
                properties=properties
            )
            await result.consume()
    
    async def get_shortest_contagion_path(
        self,
        source_id: UUID,
        target_id: UUID,
        min_probability: float = 0.1
    ) -> Optional[Dict]:
        """
        Find shortest high-probability contagion path between institutions
        
        Args:
            source_id: Source institution
            target_id: Target institution
            min_probability: Minimum contagion probability threshold
        
        Returns:
            Path information or None
        """
        query = """
        MATCH path = (source:Institution {id: $source_id})
        -[:HAS_EXPOSURE*1..5]->(target:Institution {id: $target_id})
        WHERE ALL(rel IN relationships(path) WHERE rel.contagion_probability >= $min_prob)
        WITH path,
             reduce(prob = 1.0, rel in relationships(path) | 
                prob * rel.contagion_probability) AS path_probability
        ORDER BY path_probability DESC
        LIMIT 1
        RETURN 
            [node IN nodes(path) | node.id] AS node_ids,
            path_probability,
            length(path) AS path_length
        """
        
        async with self.session() as session:
            result = await session.run(
                query,
                source_id=str(source_id),
                target_id=str(target_id),
                min_prob=min_probability
            )
            
            record = await result.single()
            if record:
                return {
                    "node_ids": record["node_ids"],
                    "probability": record["path_probability"],
                    "length": record["path_length"]
                }
            return None
    
    async def find_critical_nodes(
        self,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find most critical nodes based on betweenness centrality
        
        Args:
            top_k: Number of top nodes to return
        
        Returns:
            List of critical nodes with centrality scores
        """
        query = """
        CALL gds.betweenness.stream('financial-network')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS institution_id,
               gds.util.asNode(nodeId).name AS name,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        async with self.session() as session:
            result = await session.run(query, top_k=top_k)
            records = await result.values()
            
            return [
                {
                    "institution_id": record[0],
                    "name": record[1],
                    "betweenness_score": record[2]
                }
                for record in records
            ]
    
    async def find_cycles(
        self,
        min_cycle_length: int = 2,
        max_cycle_length: int = 5
    ) -> List[List[str]]:
        """
        Detect cycles in the exposure network
        
        Cycles indicate circular dependencies that can amplify contagion.
        
        Args:
            min_cycle_length: Minimum cycle length
            max_cycle_length: Maximum cycle length
        
        Returns:
            List of cycles (each cycle is a list of institution IDs)
        """
        query = """
        MATCH path = (n:Institution)-[:HAS_EXPOSURE*..{max_len}]->(n)
        WHERE length(path) >= {min_len}
        RETURN [node IN nodes(path) | node.id] AS cycle
        LIMIT 100
        """.format(min_len=min_cycle_length, max_len=max_cycle_length)
        
        async with self.session() as session:
            result = await session.run(query)
            records = await result.values()
            return [record[0] for record in records]
    
    async def compute_cascade_risk_score(
        self,
        institution_id: UUID,
        depth: int = 3
    ) -> float:
        """
        Compute cascade risk score for an institution
        
        Measures potential impact if this institution fails.
        
        Args:
            institution_id: Institution to analyze
            depth: Maximum cascade depth
        
        Returns:
            Risk score (0-1)
        """
        query = """
        MATCH (source:Institution {id: $inst_id})
        -[:HAS_EXPOSURE*1..{depth}]->(affected:Institution)
        WITH affected, 
             reduce(prob = 1.0, rel in relationships(path) | 
                prob * rel.contagion_probability) AS impact_prob
        RETURN COUNT(DISTINCT affected) AS affected_count,
               AVG(impact_prob) AS avg_probability,
               SUM(impact_prob) AS total_risk
        """.format(depth=depth)
        
        async with self.session() as session:
            result = await session.run(query, inst_id=str(institution_id))
            record = await result.single()
            
            if record:
                # Normalize risk score
                affected_count = record["affected_count"] or 0
                avg_prob = record["avg_probability"] or 0.0
                return min(1.0, (affected_count * avg_prob) / 100)
            return 0.0
    
    async def get_exposure_network(
        self,
        min_exposure: float = 0.0
    ) -> Dict[str, Any]:
        """
        Retrieve entire exposure network
        
        Args:
            min_exposure: Minimum exposure threshold
        
        Returns:
            Network graph data (nodes and edges)
        """
        query = """
        MATCH (n:Institution)
        OPTIONAL MATCH (n)-[e:HAS_EXPOSURE]->(m:Institution)
        WHERE e.exposure_magnitude >= $min_exposure
        RETURN 
            COLLECT(DISTINCT {
                id: n.id,
                name: n.name,
                type: n.type,
                tier: n.tier
            }) AS nodes,
            COLLECT({
                source: n.id,
                target: m.id,
                exposure_type: e.type,
                exposure_magnitude: e.exposure_magnitude,
                contagion_probability: e.contagion_probability
            }) AS edges
        """
        
        async with self.session() as session:
            result = await session.run(query, min_exposure=min_exposure)
            record = await result.single()
            
            if record:
                return {
                    "nodes": record["nodes"],
                    "edges": [e for e in record["edges"] if e["target"] is not None]
                }
            return {"nodes": [], "edges": []}
    
    async def find_communities(
        self,
        algorithm: str = "louvain"
    ) -> Dict[str, int]:
        """
        Detect communities in the financial network
        
        Args:
            algorithm: Community detection algorithm ('louvain', 'label_propagation')
        
        Returns:
            Mapping of institution_id -> community_id
        """
        if algorithm == "louvain":
            query = """
            CALL gds.louvain.stream('financial-network')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS institution_id,
                   communityId
            """
        else:  # label_propagation
            query = """
            CALL gds.labelPropagation.stream('financial-network')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS institution_id,
                   communityId
            """
        
        async with self.session() as session:
            result = await session.run(query)
            records = await result.values()
            
            return {
                record[0]: record[1]
                for record in records
            }
    
    async def create_graph_projection(
        self,
        projection_name: str = "financial-network"
    ):
        """
        Create named graph projection for use with GDS algorithms
        
        Args:
            projection_name: Name for the projection
        """
        query = """
        CALL gds.graph.project(
            $projection_name,
            'Institution',
            {
                HAS_EXPOSURE: {
                    properties: ['exposure_magnitude', 'contagion_probability']
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, projection_name=projection_name)
                record = await result.single()
                logger.info(
                    "Created graph projection",
                    name=record["graphName"],
                    nodes=record["nodeCount"],
                    edges=record["relationshipCount"]
                )
            except Exception as e:
                # Projection might already exist
                logger.warning("Graph projection exists or failed", error=str(e))
    
    async def clear_database(self):
        """Clear all data from Neo4j (use with caution!)"""
        query = "MATCH (n) DETACH DELETE n"

        async with self.session() as session:
            await session.run(query)
            logger.warning("Cleared Neo4j database")

    async def sync_network_from_simulation(
        self,
        simulation_id: str,
        network: Any,
        timestep: int
    ) -> Dict[str, Any]:
        """
        Sync NetworkX graph to Neo4j for a simulation timestep

        Args:
            simulation_id: Unique simulation identifier
            network: NetworkX DiGraph from simulation
            timestep: Current simulation timestep

        Returns:
            Sync statistics
        """
        import networkx as nx

        nodes_created = 0
        edges_created = 0

        async with self.session() as session:
            # Create nodes with simulation context
            for node_id in network.nodes():
                node_data = network.nodes[node_id]
                agent = node_data.get('agent')

                properties = {
                    'simulation_id': simulation_id,
                    'timestep': timestep,
                    'node_id': node_id,
                    'agent_type': node_data.get('agent_type', 'unknown'),
                    'alive': node_data.get('alive', True)
                }

                # Add agent-specific properties
                if agent:
                    properties['health'] = float(agent.compute_health())
                    if hasattr(agent, 'capital'):
                        properties['capital'] = float(agent.capital)
                    if hasattr(agent, 'liquidity'):
                        properties['liquidity'] = float(agent.liquidity)
                    if hasattr(agent, 'crar'):
                        properties['crar'] = float(agent.crar)

                query = """
                MERGE (n:SimNode {simulation_id: $simulation_id, node_id: $node_id, timestep: $timestep})
                SET n += $properties
                RETURN n
                """

                await session.run(
                    query,
                    simulation_id=simulation_id,
                    node_id=node_id,
                    timestep=timestep,
                    properties=properties
                )
                nodes_created += 1

            # Create edges
            for source, target, edge_data in network.edges(data=True):
                properties = {
                    'simulation_id': simulation_id,
                    'timestep': timestep,
                    'weight': float(edge_data.get('weight', 0.0)),
                    'edge_type': edge_data.get('edge_type', edge_data.get('type', 'unknown'))
                }

                query = """
                MATCH (source:SimNode {simulation_id: $simulation_id, node_id: $source, timestep: $timestep})
                MATCH (target:SimNode {simulation_id: $simulation_id, node_id: $target, timestep: $timestep})
                MERGE (source)-[r:EXPOSES {simulation_id: $simulation_id, timestep: $timestep}]->(target)
                SET r += $properties
                RETURN r
                """

                await session.run(
                    query,
                    simulation_id=simulation_id,
                    source=source,
                    target=target,
                    timestep=timestep,
                    properties=properties
                )
                edges_created += 1

        logger.info(
            f"Synced network to Neo4j",
            simulation_id=simulation_id,
            timestep=timestep,
            nodes=nodes_created,
            edges=edges_created
        )

        return {
            'nodes_created': nodes_created,
            'edges_created': edges_created,
            'timestep': timestep
        }

    async def load_network_to_simulation(
        self,
        simulation_id: str,
        timestep: int
    ) -> Dict[str, Any]:
        """
        Load Neo4j graph back to NetworkX for simulation

        Args:
            simulation_id: Unique simulation identifier
            timestep: Timestep to load

        Returns:
            Graph data (nodes and edges)
        """
        async with self.session() as session:
            # Load nodes
            nodes_query = """
            MATCH (n:SimNode {simulation_id: $simulation_id, timestep: $timestep})
            RETURN n.node_id AS node_id, properties(n) AS properties
            """

            nodes_result = await session.run(
                nodes_query,
                simulation_id=simulation_id,
                timestep=timestep
            )
            nodes = await nodes_result.values()

            # Load edges
            edges_query = """
            MATCH (source:SimNode {simulation_id: $simulation_id, timestep: $timestep})
                  -[r:EXPOSES {simulation_id: $simulation_id, timestep: $timestep}]->
                  (target:SimNode {simulation_id: $simulation_id, timestep: $timestep})
            RETURN source.node_id AS source, target.node_id AS target, properties(r) AS properties
            """

            edges_result = await session.run(
                edges_query,
                simulation_id=simulation_id,
                timestep=timestep
            )
            edges = await edges_result.values()

        return {
            'nodes': [{'id': n[0], 'properties': n[1]} for n in nodes],
            'edges': [{'source': e[0], 'target': e[1], 'properties': e[2]} for e in edges]
        }

    async def run_pagerank_algorithm(
        self,
        simulation_id: str,
        timestep: int,
        dampingFactor: float = 0.85,
        max_iterations: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Run PageRank algorithm using Neo4j GDS

        Args:
            simulation_id: Simulation identifier
            timestep: Timestep to analyze
            dampingFactor: PageRank damping factor
            max_iterations: Maximum iterations

        Returns:
            List of nodes with PageRank scores
        """
        graph_name = f"sim_{simulation_id}_t{timestep}_pagerank"

        async with self.session() as session:
            # Create graph projection
            try:
                projection_query = """
                CALL gds.graph.project(
                    $graphName,
                    {
                        SimNode: {
                            label: 'SimNode',
                            properties: ['health', 'capital']
                        }
                    },
                    {
                        EXPOSES: {
                            type: 'EXPOSES',
                            orientation: 'NATURAL',
                            properties: ['weight']
                        }
                    },
                    {
                        nodeProperties: {
                            simulation_id: $simulation_id,
                            timestep: $timestep
                        },
                        relationshipProperties: {
                            simulation_id: $simulation_id,
                            timestep: $timestep
                        }
                    }
                )
                """
                await session.run(
                    projection_query,
                    graphName=graph_name,
                    simulation_id=simulation_id,
                    timestep=timestep
                )
            except Exception as e:
                logger.debug(f"Graph projection exists or failed: {e}")

            # Run PageRank
            pagerank_query = """
            CALL gds.pageRank.stream($graphName, {
                dampingFactor: $dampingFactor,
                maxIterations: $maxIterations
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).node_id AS node_id,
                   gds.util.asNode(nodeId).agent_type AS agent_type,
                   score
            ORDER BY score DESC
            """

            result = await session.run(
                pagerank_query,
                graphName=graph_name,
                dampingFactor=dampingFactor,
                maxIterations=max_iterations
            )
            records = await result.values()

            # Clean up projection
            try:
                await session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
            except Exception as e:
                logger.debug(f"Failed to drop graph projection: {e}")

        return [
            {
                'node_id': record[0],
                'agent_type': record[1],
                'pagerank_score': float(record[2])
            }
            for record in records
        ]

    async def run_louvain_community_detection(
        self,
        simulation_id: str,
        timestep: int,
        max_levels: int = 10
    ) -> Dict[str, Any]:
        """
        Run Louvain community detection using Neo4j GDS

        Args:
            simulation_id: Simulation identifier
            timestep: Timestep to analyze
            max_levels: Maximum hierarchy levels

        Returns:
            Community assignments and statistics
        """
        graph_name = f"sim_{simulation_id}_t{timestep}_louvain"

        async with self.session() as session:
            # Create graph projection
            try:
                projection_query = """
                CALL gds.graph.project(
                    $graphName,
                    'SimNode',
                    'EXPOSES',
                    {
                        nodeProperties: {
                            filter: {
                                simulation_id: $simulation_id,
                                timestep: $timestep
                            }
                        },
                        relationshipProperties: ['weight']
                    }
                )
                """
                await session.run(
                    projection_query,
                    graphName=graph_name,
                    simulation_id=simulation_id,
                    timestep=timestep
                )
            except Exception as e:
                logger.debug(f"Graph projection exists or failed: {e}")

            # Run Louvain
            louvain_query = """
            CALL gds.louvain.stream($graphName, {
                maxLevels: $maxLevels,
                relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, communityId, intermediateCommunityIds
            RETURN gds.util.asNode(nodeId).node_id AS node_id,
                   gds.util.asNode(nodeId).agent_type AS agent_type,
                   communityId,
                   intermediateCommunityIds
            """

            result = await session.run(
                louvain_query,
                graphName=graph_name,
                maxLevels=max_levels
            )
            records = await result.values()

            # Clean up
            try:
                await session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
            except Exception as e:
                logger.debug(f"Failed to drop graph projection: {e}")

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
                {
                    'node_id': record[0],
                    'community_id': int(record[2])
                }
                for record in records
            ]
        }

    async def run_shortest_path_analysis(
        self,
        simulation_id: str,
        timestep: int,
        source_node: str,
        target_node: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest weighted path between two nodes

        Args:
            simulation_id: Simulation identifier
            timestep: Timestep to analyze
            source_node: Source node ID
            target_node: Target node ID

        Returns:
            Path information or None if no path exists
        """
        async with self.session() as session:
            query = """
            MATCH (source:SimNode {simulation_id: $simulation_id, timestep: $timestep, node_id: $source_node})
            MATCH (target:SimNode {simulation_id: $simulation_id, timestep: $timestep, node_id: $target_node})
            MATCH path = shortestPath((source)-[:EXPOSES*]-(target))
            WITH path,
                 reduce(w = 0, rel in relationships(path) | w + rel.weight) AS total_weight
            RETURN [node IN nodes(path) | node.node_id] AS path_nodes,
                   length(path) AS path_length,
                   total_weight
            ORDER BY total_weight DESC
            LIMIT 1
            """

            result = await session.run(
                query,
                simulation_id=simulation_id,
                timestep=timestep,
                source_node=source_node,
                target_node=target_node
            )

            record = await result.single()
            if record:
                return {
                    'path': record['path_nodes'],
                    'length': record['path_length'],
                    'total_weight': float(record['total_weight'])
                }
            return None


# Global client instance
neo4j_client = Neo4jClient()


async def init_neo4j():
    """Initialize Neo4j connection"""
    await neo4j_client.connect()


async def close_neo4j():
    """Close Neo4j connection"""
    await neo4j_client.close()
