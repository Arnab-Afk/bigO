"""
Network Synchronization Service

Handles syncing simulation network state to Neo4j for persistent storage
and advanced graph analysis.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.db.neo4j import neo4j_client
from app.core.logging import logger


class NetworkSyncService:
    """
    Service for synchronizing simulation network state to Neo4j

    Provides methods for:
    - Incremental sync after each timestep
    - Bulk sync of full simulation history
    - Cleanup operations
    """

    def __init__(self):
        """Initialize the network sync service"""
        self.client = neo4j_client

    async def sync_snapshot_to_neo4j(
        self,
        simulation_id: str,
        network: Any,
        timestep: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sync a single simulation timestep snapshot to Neo4j

        Args:
            simulation_id: Unique simulation identifier
            network: NetworkX DiGraph from simulation
            timestep: Current timestep
            metadata: Optional metadata about this snapshot

        Returns:
            Sync statistics
        """
        try:
            # Sync the network
            sync_stats = await self.client.sync_network_from_simulation(
                simulation_id=simulation_id,
                network=network,
                timestep=timestep
            )

            # Store metadata if provided
            if metadata:
                await self._store_snapshot_metadata(
                    simulation_id=simulation_id,
                    timestep=timestep,
                    metadata=metadata
                )

            logger.info(
                "Synced snapshot to Neo4j",
                simulation_id=simulation_id,
                timestep=timestep,
                nodes=sync_stats['nodes_created'],
                edges=sync_stats['edges_created']
            )

            return {
                'success': True,
                'simulation_id': simulation_id,
                'timestep': timestep,
                'stats': sync_stats
            }

        except Exception as e:
            logger.error(
                "Failed to sync snapshot to Neo4j",
                simulation_id=simulation_id,
                timestep=timestep,
                error=str(e)
            )
            return {
                'success': False,
                'error': str(e)
            }

    async def bulk_sync_history(
        self,
        simulation_id: str,
        snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Bulk sync full simulation history to Neo4j

        Args:
            simulation_id: Unique simulation identifier
            snapshots: List of simulation snapshots with network state

        Returns:
            Sync statistics
        """
        total_nodes = 0
        total_edges = 0
        failed_timesteps = []

        logger.info(
            "Starting bulk sync to Neo4j",
            simulation_id=simulation_id,
            snapshot_count=len(snapshots)
        )

        for snapshot in snapshots:
            try:
                timestep = snapshot['timestep']
                network_state = snapshot['network_state']

                # Reconstruct network from snapshot
                import networkx as nx
                network = nx.DiGraph()

                # Add nodes
                for node_data in network_state['nodes']:
                    network.add_node(
                        node_data['id'],
                        agent_type=node_data.get('type'),
                        alive=node_data.get('alive'),
                        health=node_data.get('health')
                    )

                # Add edges
                for edge_data in network_state['edges']:
                    network.add_edge(
                        edge_data['source'],
                        edge_data['target'],
                        weight=edge_data.get('weight', 0),
                        edge_type=edge_data.get('type', 'unknown')
                    )

                # Sync this snapshot
                sync_stats = await self.client.sync_network_from_simulation(
                    simulation_id=simulation_id,
                    network=network,
                    timestep=timestep
                )

                total_nodes += sync_stats['nodes_created']
                total_edges += sync_stats['edges_created']

            except Exception as e:
                logger.error(
                    "Failed to sync timestep",
                    timestep=timestep,
                    error=str(e)
                )
                failed_timesteps.append(timestep)

        logger.info(
            "Completed bulk sync to Neo4j",
            simulation_id=simulation_id,
            total_nodes=total_nodes,
            total_edges=total_edges,
            failed_count=len(failed_timesteps)
        )

        return {
            'success': len(failed_timesteps) == 0,
            'simulation_id': simulation_id,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'snapshots_processed': len(snapshots),
            'failed_timesteps': failed_timesteps
        }

    async def clear_simulation_graph(
        self,
        simulation_id: str,
        timestep: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Clear simulation data from Neo4j

        Args:
            simulation_id: Simulation identifier
            timestep: Optional specific timestep to clear (if None, clears all)

        Returns:
            Deletion statistics
        """
        async with self.client.session() as session:
            if timestep is not None:
                # Clear specific timestep
                query = """
                MATCH (n:SimNode {simulation_id: $simulation_id, timestep: $timestep})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                result = await session.run(
                    query,
                    simulation_id=simulation_id,
                    timestep=timestep
                )
            else:
                # Clear all timesteps for this simulation
                query = """
                MATCH (n:SimNode {simulation_id: $simulation_id})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                result = await session.run(query, simulation_id=simulation_id)

            record = await result.single()
            deleted_count = record['deleted_count'] if record else 0

        logger.info(
            "Cleared simulation graph",
            simulation_id=simulation_id,
            timestep=timestep,
            deleted_nodes=deleted_count
        )

        return {
            'simulation_id': simulation_id,
            'timestep': timestep,
            'deleted_nodes': deleted_count
        }

    async def _store_snapshot_metadata(
        self,
        simulation_id: str,
        timestep: int,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store metadata about a snapshot

        Args:
            simulation_id: Simulation identifier
            timestep: Timestep
            metadata: Metadata dictionary
        """
        async with self.client.session() as session:
            query = """
            MERGE (m:SimMetadata {simulation_id: $simulation_id, timestep: $timestep})
            SET m += $metadata,
                m.synced_at = datetime()
            """
            await session.run(
                query,
                simulation_id=simulation_id,
                timestep=timestep,
                metadata=metadata
            )

    async def get_sync_status(
        self,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get sync status for a simulation

        Args:
            simulation_id: Simulation identifier

        Returns:
            Sync status information
        """
        async with self.client.session() as session:
            # Count synced timesteps
            query = """
            MATCH (n:SimNode {simulation_id: $simulation_id})
            WITH DISTINCT n.timestep AS timestep
            RETURN collect(timestep) AS timesteps,
                   count(timestep) AS timestep_count
            """
            result = await session.run(query, simulation_id=simulation_id)
            record = await result.single()

            if not record:
                return {
                    'simulation_id': simulation_id,
                    'is_synced': False,
                    'timesteps_synced': 0,
                    'timesteps': []
                }

            timesteps = sorted(record['timesteps'])

            return {
                'simulation_id': simulation_id,
                'is_synced': True,
                'timesteps_synced': record['timestep_count'],
                'timesteps': timesteps,
                'first_timestep': timesteps[0] if timesteps else None,
                'last_timestep': timesteps[-1] if timesteps else None
            }

    async def load_snapshot_from_neo4j(
        self,
        simulation_id: str,
        timestep: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load a specific snapshot from Neo4j back to memory

        Args:
            simulation_id: Simulation identifier
            timestep: Timestep to load

        Returns:
            Snapshot data or None if not found
        """
        try:
            graph_data = await self.client.load_network_to_simulation(
                simulation_id=simulation_id,
                timestep=timestep
            )

            return {
                'simulation_id': simulation_id,
                'timestep': timestep,
                'nodes': graph_data['nodes'],
                'edges': graph_data['edges']
            }

        except Exception as e:
            logger.error(
                "Failed to load snapshot from Neo4j",
                simulation_id=simulation_id,
                timestep=timestep,
                error=str(e)
            )
            return None
