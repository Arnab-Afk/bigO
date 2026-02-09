"""
LLM Chat API Endpoints
======================
API endpoints for LLM-powered chatbot and simulation explanations.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import structlog

from app.services.llm_service import get_llm_service
from app.api.v1.abm_simulation import ACTIVE_SIMULATIONS

logger = structlog.get_logger()
router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chatbot response"""
    simulation_id: str = Field(..., description="Simulation ID for context")
    message: str = Field(..., description="User's message")
    history: List[ChatMessage] = Field(default=[], description="Chat history")


class ChatResponse(BaseModel):
    """Response from chatbot"""
    message: str = Field(..., description="Assistant's response")
    timestamp: str = Field(..., description="Response timestamp")


class ExplainEventRequest(BaseModel):
    """Request to explain a simulation event"""
    simulation_id: str = Field(..., description="Simulation ID")
    event_type: str = Field(..., description="Event type (simulation_step, entity_default, etc.)")
    event_data: Dict[str, Any] = Field(..., description="Event data")


class ExplainEventResponse(BaseModel):
    """Response with event explanation"""
    explanation: str = Field(..., description="Natural language explanation")
    event_type: str = Field(..., description="Event type")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest) -> ChatResponse:
    """
    Chat with LLM assistant about simulation.
    
    The assistant can explain what's happening, answer questions,
    and provide insights about the financial network simulation.
    
    Example questions:
    - "Why did Bank_025 fail?"
    - "What does system health below 50% mean?"
    - "How can I prevent contagion?"
    - "Explain the last shock event"
    """
    try:
        llm_service = get_llm_service()
        
        # Get simulation context if available
        simulation_context = None
        if request.simulation_id in ACTIVE_SIMULATIONS:
            ecosystem = ACTIVE_SIMULATIONS[request.simulation_id]
            simulation_context = {
                "timestep": ecosystem.timestep,
                "system_health": ecosystem.global_state.get("system_health", 0) * 100,
                "alive_agents": len([a for a in ecosystem.agents.values() if a.alive]),
                "total_agents": len(ecosystem.agents),
            }
        
        # Convert chat history to format expected by LLM service
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        # Generate response
        response_text = await llm_service.generate_chat_response(
            user_message=request.message,
            chat_history=history,
            simulation_context=simulation_context
        )
        
        from datetime import datetime
        return ChatResponse(
            message=response_text,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error("chat_endpoint_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chat response: {str(e)}"
        )


@router.post("/explain", response_model=ExplainEventResponse)
async def explain_event(request: ExplainEventRequest) -> ExplainEventResponse:
    """
    Generate natural language explanation for a simulation event.
    
    This endpoint is called automatically by the frontend to explain
    real-time events as they occur in the simulation.
    
    Supported event types:
    - simulation_step: Timestep completion
    - entity_default: Entity failure
    - shock_applied: External shock
    - pending_decision: User decision required
    - health_warning: System health alert
    """
    try:
        llm_service = get_llm_service()
        
        # Get additional context if available
        context = None
        if request.simulation_id in ACTIVE_SIMULATIONS:
            ecosystem = ACTIVE_SIMULATIONS[request.simulation_id]
            context = {
                "global_state": ecosystem.global_state,
                "timestep": ecosystem.timestep
            }
        
        # Generate explanation
        explanation = await llm_service.generate_simulation_explanation(
            event_type=request.event_type,
            event_data=request.event_data,
            context=context
        )
        
        return ExplainEventResponse(
            explanation=explanation,
            event_type=request.event_type
        )
    
    except Exception as e:
        logger.error("explain_endpoint_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@router.post("/narrate-step")
async def narrate_simulation_step(
    simulation_id: str = Body(...),
    timestep: int = Body(...),
    snapshot: Dict[str, Any] = Body(...)
):
    """
    Generate narrative description of a simulation timestep.
    
    This can be called after each step to get a running commentary
    of what's happening in the simulation.
    """
    try:
        llm_service = get_llm_service()
        
        event_data = {
            "timestep": timestep,
            "system_health": snapshot.get("global_metrics", {}).get("system_health", 0),
            "alive_agents": snapshot.get("global_metrics", {}).get("alive_agents", 0),
            "total_agents": snapshot.get("global_metrics", {}).get("total_agents", 0)
        }
        
        explanation = await llm_service.generate_simulation_explanation(
            event_type="simulation_step",
            event_data=event_data
        )
        
        return {
            "narration": explanation,
            "timestep": timestep
        }
    
    except Exception as e:
        logger.error("narrate_step_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate narration: {str(e)}"
        )


@router.get("/health")
async def llm_health_check():
    """
    Check if LLM service is configured and available.
    """
    try:
        llm_service = get_llm_service()
        has_api_key = llm_service.api_key is not None
        
        return {
            "status": "configured" if has_api_key else "not_configured",
            "model": llm_service.default_model,
            "message": "LLM service ready" if has_api_key else "API key not set (set FEATHERLESS_API_KEY env var)"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
