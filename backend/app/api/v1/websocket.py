"""WebSocket endpoints for real-time simulation updates"""
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()

router = APIRouter()

# Store active WebSocket connections per simulation
active_connections: Dict[str, list[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for simulations"""

    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, simulation_id: str):
        """Accept a new WebSocket connection for a simulation"""
        await websocket.accept()
        if simulation_id not in self.active_connections:
            self.active_connections[simulation_id] = []
        self.active_connections[simulation_id].append(websocket)
        logger.info(
            "websocket_connected",
            simulation_id=simulation_id,
            total_connections=len(self.active_connections[simulation_id]),
        )

    def disconnect(self, websocket: WebSocket, simulation_id: str):
        """Remove a WebSocket connection"""
        if simulation_id in self.active_connections:
            if websocket in self.active_connections[simulation_id]:
                self.active_connections[simulation_id].remove(websocket)
                logger.info(
                    "websocket_disconnected",
                    simulation_id=simulation_id,
                    remaining_connections=len(self.active_connections[simulation_id]),
                )
            if not self.active_connections[simulation_id]:
                del self.active_connections[simulation_id]

    async def send_message(self, simulation_id: str, message: dict):
        """Send a message to all connected clients for a simulation"""
        if simulation_id not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[simulation_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(
                    "websocket_send_error",
                    simulation_id=simulation_id,
                    error=str(e),
                )
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection, simulation_id)

    async def broadcast(self, simulation_id: str, event_type: str, data: dict):
        """Broadcast an event to all connected clients"""
        message = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        await self.send_message(simulation_id, message)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/simulation/{simulation_id}")
async def simulation_websocket(websocket: WebSocket, simulation_id: str):
    """
    WebSocket endpoint for real-time simulation updates
    
    Events pushed to clients:
    - simulation_step: New timestep completed
    - entity_default: An entity has defaulted
    - shock_applied: External shock applied to system
    - health_warning: System health below threshold
    - pending_decision: User decision required
    """
    await manager.connect(websocket, simulation_id)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "simulation_id": simulation_id,
                "message": "Connected to simulation stream"
            }
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client (ping/pong for keepalive)
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0  # 30 second timeout
                )
                
                # Handle client messages
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, simulation_id)
        logger.info("websocket_client_disconnected", simulation_id=simulation_id)
    except Exception as e:
        logger.error(
            "websocket_error",
            simulation_id=simulation_id,
            error=str(e),
            exc_info=True,
        )
        manager.disconnect(websocket, simulation_id)


@router.get("/ws/test")
async def websocket_test_page():
    """Test page for WebSocket functionality"""
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>WebSocket Test</title>
            <style>
                body { font-family: monospace; padding: 20px; }
                #messages { 
                    border: 1px solid #ccc; 
                    padding: 10px; 
                    height: 400px; 
                    overflow-y: auto;
                    background: #f5f5f5;
                }
                .message { 
                    margin: 5px 0; 
                    padding: 5px;
                    background: white;
                    border-left: 3px solid #0066cc;
                }
                .connected { border-left-color: #00cc66; }
                .error { border-left-color: #cc0000; }
                button { 
                    padding: 10px 20px; 
                    margin: 10px 5px 10px 0;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <h1>WebSocket Test Client</h1>
            <input type="text" id="simId" placeholder="Simulation ID" style="width: 400px; padding: 10px;"/>
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <button onclick="sendPing()">Send Ping</button>
            <button onclick="clearMessages()">Clear</button>
            <div id="status">Status: Disconnected</div>
            <div id="messages"></div>

            <script>
                let ws = null;
                const messagesDiv = document.getElementById('messages');
                const statusDiv = document.getElementById('status');

                function connect() {
                    const simId = document.getElementById('simId').value || 'test-sim-123';
                    const wsUrl = `ws://localhost:17170/api/v1/ws/simulation/${simId}`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = () => {
                        statusDiv.textContent = 'Status: Connected';
                        statusDiv.style.color = 'green';
                        addMessage('Connected to simulation: ' + simId, 'connected');
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        addMessage(JSON.stringify(data, null, 2), 'message');
                    };
                    
                    ws.onerror = (error) => {
                        addMessage('Error: ' + error, 'error');
                    };
                    
                    ws.onclose = () => {
                        statusDiv.textContent = 'Status: Disconnected';
                        statusDiv.style.color = 'red';
                        addMessage('Disconnected', 'error');
                    };
                }

                function disconnect() {
                    if (ws) {
                        ws.close();
                        ws = null;
                    }
                }

                function sendPing() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                        addMessage('Sent: ping', 'message');
                    }
                }

                function addMessage(text, className = 'message') {
                    const div = document.createElement('div');
                    div.className = 'message ' + className;
                    div.textContent = new Date().toLocaleTimeString() + ' - ' + text;
                    messagesDiv.appendChild(div);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }

                function clearMessages() {
                    messagesDiv.innerHTML = '';
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


# Export manager for use in other modules
__all__ = ["router", "manager"]
