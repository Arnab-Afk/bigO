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


# Global client instance
neo4j_client = Neo4jClient()


async def init_neo4j():
    """Initialize Neo4j connection"""
    await neo4j_client.connect()


async def close_neo4j():
    """Close Neo4j connection"""
    await neo4j_client.close()
