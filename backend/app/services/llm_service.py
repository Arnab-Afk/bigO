"""
LLM Service - Featherless AI Integration
==========================================
Provides natural language explanations for simulation events using Featherless AI API.
"""

import httpx
import structlog
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.core.config import settings

logger = structlog.get_logger()


class FeatherlessAIService:
    """
    Service for generating natural language explanations using Featherless AI.
    Supports streaming and non-streaming responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Featherless AI service.
        
        Args:
            api_key: Featherless API key (or loaded from settings)
            base_url: API base URL (or loaded from settings)
            model: Model to use (or loaded from settings)
        """
        self.api_key = api_key or settings.FEATHERLESS_API_KEY
        self.base_url = base_url or settings.FEATHERLESS_API_BASE
        self.default_model = model or settings.FEATHERLESS_MODEL
    
    async def generate_simulation_explanation(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 300
    ) -> str:
        """
        Generate natural language explanation for a simulation event.
        
        Args:
            event_type: Type of event (simulation_step, entity_default, shock_applied, etc.)
            event_data: Event data from WebSocket
            context: Additional context (system metrics, history, etc.)
            max_tokens: Maximum tokens in response
            
        Returns:
            Natural language explanation string
        """
        prompt = self._create_prompt(event_type, event_data, context)
        
        try:
            response = await self._call_api(
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            explanation = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(
                "llm_explanation_generated",
                event_type=event_type,
                explanation_length=len(explanation)
            )
            return explanation.strip()
            
        except Exception as e:
            logger.error("llm_explanation_failed", error=str(e), event_type=event_type)
            return self._fallback_explanation(event_type, event_data)
    
    async def generate_chat_response(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        simulation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate chatbot response for user questions about simulation.
        
        Args:
            user_message: User's question
            chat_history: Previous conversation history
            simulation_context: Current simulation state
            
        Returns:
            Chatbot response
        """
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(simulation_context)
            }
        ]
        
        # Add chat history
        messages.extend(chat_history[-10:])  # Last 10 messages for context
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = await self._call_api(
                messages=messages,
                max_tokens=500,
                temperature=0.8
            )
            
            reply = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return reply.strip()
            
        except Exception as e:
            logger.error("chat_response_failed", error=str(e))
            return "I'm having trouble processing your question right now. Please try again."
    
    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Call Featherless AI API.
        
        Args:
            messages: Chat messages
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            API response
        """
        if not self.api_key:
            raise ValueError("Featherless API key not configured")
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    
    def _get_system_prompt(self, simulation_context: Optional[Dict[str, Any]] = None) -> str:
        """Get system prompt for LLM"""
        base_prompt = """You are an AI assistant for RUDRA (Resilient Unified Decision & Risk Analytics), a financial network simulation platform.

Your role is to explain complex financial simulations in simple, clear language that anyone can understand.

Key concepts:
- **System Health**: Overall stability of the financial network (0-100%)
- **Entities**: Banks, CCPs (clearing houses), regulators in the network
- **Default**: When an entity runs out of capital and fails
- **Contagion**: When one entity's failure causes others to fail (domino effect)
- **Shock**: External event that impacts the system (market crash, liquidity crisis)
- **Risk Appetite**: How aggressively an entity takes risks
- **NPA**: Non-Performing Assets - bad loans that won't be repaid
- **Liquidity**: Cash available to meet immediate obligations

Guidelines:
- Use simple language, avoid jargon
- Use analogies (e.g., "like a house of cards")
- Be concise but informative
- Focus on what happened and why it matters
- Suggest actions when relevant
- Use emojis sparingly for clarity (⚠️ 📉 📈 ✅ ❌)
"""
        
        if simulation_context:
            context_info = f"""
Current Simulation State:
- Timestep: {simulation_context.get('timestep', 'N/A')}
- System Health: {simulation_context.get('system_health', 'N/A')}%
- Active Entities: {simulation_context.get('alive_agents', 'N/A')}/{simulation_context.get('total_agents', 'N/A')}
"""
            return base_prompt + context_info
        
        return base_prompt
    
    def _create_prompt(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create prompt for specific event type"""
        
        if event_type == "simulation_step":
            timestep = event_data.get("timestep", "?")
            health = event_data.get("system_health", 0) * 100
            alive = event_data.get("alive_agents", 0)
            
            return f"""Explain this simulation update in 2-3 sentences:

Timestep {timestep} completed:
- System Health: {health:.1f}%
- Active Entities: {alive}

What happened and why does it matter?"""

        elif event_type == "entity_default":
            entity_id = event_data.get("entity_id", "Unknown")
            entity_type = event_data.get("entity_type", "entity")
            
            return f"""Explain this entity failure in 2-3 sentences:

{entity_type.upper()} '{entity_id}' has defaulted.

What caused this and what are the implications?"""
        
        elif event_type == "shock_applied":
            shock_type = event_data.get("shock_type", "unknown")
            target = event_data.get("target", "system")
            magnitude = event_data.get("magnitude", 0)
            
            return f"""Explain this external shock in 2-3 sentences:

Shock Type: {shock_type}
Target: {target}
Magnitude: {magnitude:.1%}

What does this mean for the financial system?"""
        
        elif event_type == "pending_decision":
            title = event_data.get("title", "Decision Required")
            description = event_data.get("description", "")
            
            return f"""Explain this risk alert in 2-3 sentences:

{title}
{description}

What should the user consider?"""
        
        elif event_type == "health_warning":
            health = event_data.get("system_health", 0) * 100
            
            return f"""System health has dropped to {health:.1f}%. Explain what this means and what might happen next in 2-3 sentences."""
        
        else:
            return f"Explain this simulation event in simple terms: {event_type}"
    
    def _fallback_explanation(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Fallback explanation if LLM fails"""
        
        fallbacks = {
            "simulation_step": f"⏱️ Timestep {event_data.get('timestep', '?')} completed. System health at {event_data.get('system_health', 0)*100:.1f}%.",
            "entity_default": f"❌ {event_data.get('entity_id', 'Entity')} has failed and exited the network.",
            "shock_applied": f"⚡ {event_data.get('shock_type', 'External shock')} applied to the system.",
            "pending_decision": f"⚠️ Your action is required: {event_data.get('title', 'Decision needed')}",
            "health_warning": "⚠️ System health is declining. Monitor the situation closely."
        }
        
        return fallbacks.get(event_type, f"Event: {event_type}")


# Singleton instance
_llm_service: Optional[FeatherlessAIService] = None


def get_llm_service() -> FeatherlessAIService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = FeatherlessAIService()
    return _llm_service
