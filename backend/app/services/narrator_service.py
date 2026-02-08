"""
Auto-Narrator Service
=====================
Automatically generates LLM explanations for simulation events.
"""

from app.services.llm_service import get_llm_service
from app.api.v1.websocket import manager as ws_manager
import structlog
import asyncio

logger = structlog.get_logger()


async def narrate_event(
    simulation_id: str,
    event_type: str,
    event_data: dict,
    context: dict = None
):
    """
    Generate LLM explanation for an event and broadcast it via WebSocket.
    
    This function is called automatically when significant simulation events occur.
    It generates a natural language explanation and sends it to all connected clients.
    
    Args:
        simulation_id: ID of the simulation
        event_type: Type of event (simulation_step, entity_default, etc.)
        event_data: Event-specific data
        context: Additional context (global state, etc.)
    """
    try:
        llm_service = get_llm_service()
        
        # Generate explanation asynchronously
        explanation = await llm_service.generate_simulation_explanation(
            event_type=event_type,
            event_data=event_data,
            context=context
        )
        
        # Broadcast narration to all connected clients
        await ws_manager.broadcast(
            simulation_id=simulation_id,
            message={
                "type": "llm_narration",
                "data": {
                    "event_type": event_type,
                    "narration": explanation,
                    "timestamp": event_data.get("timestamp", "")
                }
            }
        )
        
        logger.info(
            "narration_generated",
            simulation_id=simulation_id,
            event_type=event_type,
            narration_length=len(explanation)
        )
    
    except Exception as e:
        logger.error(
            "narration_error",
            simulation_id=simulation_id,
            event_type=event_type,
            error=str(e),
            exc_info=True
        )


def narrate_event_background(
    simulation_id: str,
    event_type: str,
    event_data: dict,
    context: dict = None
):
    """
    Fire-and-forget narration - doesn't block the simulation.
    
    Creates a background task to generate and broadcast narration
    without waiting for completion.
    """
    try:
        # Get or create event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create task
            asyncio.create_task(
                narrate_event(
                    simulation_id=simulation_id,
                    event_type=event_type,
                    event_data=event_data,
                    context=context
                )
            )
        else:
            # If no loop is running, use ensure_future
            asyncio.ensure_future(
                narrate_event(
                    simulation_id=simulation_id,
                    event_type=event_type,
                    event_data=event_data,
                    context=context
                )
            )
    except Exception as e:
        logger.error("failed_to_create_narration_task", error=str(e))
