# LLM Chatbot Integration - Complete

## Overview
Integrated Featherless AI-powered chatbot to provide natural language explanations of simulation events in real-time. The chatbot acts as an AI narrator, automatically explaining what's happening during the simulation and answering user questions about the financial network.

## Architecture

### Backend Components

#### 1. LLM Service (`backend/app/services/llm_service.py`)
- **FeatherlessAIService** class provides interface to Featherless AI API
- Uses **meta-llama/Meta-Llama-3.1-70B-Instruct** model by default
- Two main functions:
  - `generate_simulation_explanation()` - Generates event-specific explanations
  - `generate_chat_response()` - Handles interactive Q&A with context awareness

**Configuration:**
```bash
# Set in .env or environment
FEATHERLESS_API_KEY=your_api_key_here
FEATHERLESS_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct  # optional
```

**Event Types Supported:**
- `simulation_step` - Explains timestep progression and system health changes
- `entity_default` - Explains why an entity failed
- `shock_applied` - Describes impact of external shocks
- `pending_decision` - Guides user through strategic decisions
- `health_warning` - Alerts about system instability

#### 2. API Endpoints (`backend/app/api/v1/llm_chat.py`)

##### POST `/api/v1/llm/chat`
Interactive chatbot conversation.

**Request:**
```json
{
  "simulation_id": "uuid",
  "message": "Why did Bank_025 fail?",
  "history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ]
}
```

**Response:**
```json
{
  "message": "Bank_025 failed due to cascading credit defaults...",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

##### POST `/api/v1/llm/explain`
One-shot event explanation.

**Request:**
```json
{
  "simulation_id": "uuid",
  "event_type": "entity_default",
  "event_data": {
    "entity_id": "Bank_025",
    "entity_type": "BANK",
    "timestep": 15
  }
}
```

**Response:**
```json
{
  "explanation": "At timestep 15, Bank_025 defaulted...",
  "event_type": "entity_default"
}
```

##### GET `/api/v1/llm/health`
Check LLM service status.

**Response:**
```json
{
  "status": "configured",
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "message": "LLM service ready"
}
```

#### 3. Auto-Narrator Service (`backend/app/services/narrator_service.py`)
- Automatically generates LLM explanations for simulation events
- Broadcasts narrations via WebSocket to all connected clients
- Fire-and-forget pattern - doesn't block simulation execution
- Integrated into simulation step/shock/decision endpoints

**Integration Points in `abm_simulation.py`:**
```python
# After broadcasting a simulation step
if narrate_event_background:
    narrate_event_background(
        simulation_id=sim_id,
        event_type="simulation_step",
        event_data={...},
        context={...}
    )
```

### Frontend Components

#### 1. ChatBot Component (`frontend/components/ChatBot.tsx`)
Modern chatbot UI with:
- **Collapsible chat window** (minimize/maximize/close)
- **Message history** with user/assistant/system messages
- **Real-time narrations** - Auto-displays LLM-generated event explanations
- **Quick action buttons** - Pre-filled questions for common queries
- **Typing indicators** - Shows when LLM is thinking
- **Auto-scroll** - Keeps latest messages visible
- **Floating toggle button** - Easy access from anywhere

**Features:**
- System messages display AI narrator updates with 🤖 icon
- User messages in blue (right-aligned)
- Assistant messages in gray (left-aligned)
- Timestamps on all messages
- Send button + Enter key support (Shift+Enter for new line)

#### 2. Integration in ML Dashboard (`frontend/app/ml-dashboard/page.tsx`)
- WebSocket listener handles `llm_narration` message type
- Narrations stored in state array
- Passed to ChatBot component as props
- Auto-opens chat when narration arrives (optional)

**WebSocket Message Handler:**
```typescript
case 'llm_narration':
  console.log(`🤖 AI Narration: ${message.data.narration}`);
  setNarrations((prev) => [
    ...prev,
    {
      eventType: message.data.event_type,
      narration: message.data.narration,
      timestamp: message.data.timestamp
    }
  ]);
  break;
```

## Data Flow

### Auto-Narration Flow
```
1. Simulation Event Occurs (e.g., entity default)
   ↓
2. abm_simulation.py broadcasts WebSocket message
   ↓
3. narrator_service.narrate_event_background() called
   ↓
4. llm_service.generate_simulation_explanation()
   ↓
5. Featherless AI API call (async)
   ↓
6. Broadcast "llm_narration" via WebSocket
   ↓
7. Frontend receives narration
   ↓
8. ChatBot displays as system message
```

### Interactive Chat Flow
```
1. User types question in ChatBot
   ↓
2. POST /api/v1/llm/chat with message + history
   ↓
3. llm_service.generate_chat_response()
   ↓
4. Includes simulation context (health, timestep, etc.)
   ↓
5. Featherless AI generates contextual answer
   ↓
6. ChatBot displays assistant response
   ↓
7. Message added to history for context
```

## Usage Examples

### Starting a Simulation with LLM Narrator

```bash
# Backend (Terminal 1)
cd backend
source .venv/bin/activate
export FEATHERLESS_API_KEY="your_key_here"
python -m uvicorn app.main:app --host 0.0.0.0 --port 17170 --reload

# Frontend (Terminal 2)
cd frontend
npm run dev
```

### Chatbot Quick Actions
Pre-filled questions available via buttons:
- **"What is the current system health?"** - Get health summary
- **"Explain the last event"** - Understand what just happened
- **"What should I do next?"** - Get strategic recommendations

### Custom Questions
Ask anything about the simulation:
- "Why did Bank_025 fail?"
- "How is contagion spreading?"
- "Should I increase liquidity requirements?"
- "What happens if I reject this decision?"
- "Explain the sector crisis shock"

## Configuration

### Environment Variables
```bash
# Required
FEATHERLESS_API_KEY=your_api_key_here

# Optional
FEATHERLESS_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
FEATHERLESS_API_BASE=https://api.featherless.ai/v1
```

### System Prompt Customization
Edit `backend/app/services/llm_service.py`:
```python
self.system_prompt = """
You are an expert financial risk analyst...
[customize this to change AI personality/knowledge]
"""
```

### Event-Specific Prompts
Each event type has a template in `llm_service.py`:
```python
self.event_prompts = {
    "simulation_step": "Explain what happened in timestep {timestep}...",
    "entity_default": "Explain why {entity_id} failed...",
    # Add custom event types here
}
```

## Testing

### Test LLM Service Availability
```bash
curl http://localhost:17170/api/v1/llm/health
```

### Test Manual Chat
```bash
curl -X POST http://localhost:17170/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "test-sim",
    "message": "What is system health?",
    "history": []
  }'
```

### Test Event Explanation
```bash
curl -X POST http://localhost:17170/api/v1/llm/explain \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "test-sim",
    "event_type": "simulation_step",
    "event_data": {
      "timestep": 10,
      "system_health": 0.75,
      "alive_agents": 45,
      "total_agents": 50
    }
  }'
```

### Frontend Testing
1. Initialize simulation in ml-dashboard
2. Click chat icon (bottom-right)
3. Run simulation and watch auto-narrations appear
4. Ask questions in chat input
5. Try quick action buttons

## Performance Considerations

### API Rate Limits
Featherless AI has rate limits. Auto-narrator uses fire-and-forget pattern to avoid blocking simulation.

**Optimization strategies:**
- Narrations only for significant events (not every timestep)
- Batch multiple events into single explanation
- Cache similar explanations

### Cost Management
LLM API calls have costs. Control via:
- Enable/disable auto-narration: Set `ENABLE_AUTO_NARRATION=false`
- Reduce narration frequency: Only narrate major events
- Use cheaper models: Set `FEATHERLESS_MODEL=meta-llama/Llama-3.2-8B-Instruct`

### Fallback Behavior
If LLM service fails:
- Chatbot shows: "Sorry, I encountered an error. Please try again."
- Auto-narrator logs error, simulation continues normally
- Pre-written fallback explanations available in `llm_service.py`

## Troubleshooting

### "API key not set" Error
```bash
# Check environment
echo $FEATHERLESS_API_KEY

# Set in current session
export FEATHERLESS_API_KEY="your_key"

# Or in .env file
echo "FEATHERLESS_API_KEY=your_key" >> backend/.env
```

### Narrations Not Appearing
1. Check WebSocket connection (green "Live" badge)
2. Open browser console - look for `🤖 AI Narration:` logs
3. Verify backend logs show "narration_generated"
4. Test `/api/v1/llm/health` endpoint

### Slow Responses
- LLM inference can take 2-10 seconds
- Typing indicator shows while waiting
- Consider using faster model (trade-off: less accurate)

### CORS Issues
If frontend can't reach backend:
```python
# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## API Documentation
Full interactive docs available at:
- Swagger UI: http://localhost:17170/docs
- Navigate to "LLM Chat" section for endpoints

## Future Enhancements

### Planned Features
- [ ] Voice narration (text-to-speech)
- [ ] Multi-language support
- [ ] Sentiment analysis of simulation state
- [ ] Proactive recommendations
- [ ] Chat export/save functionality
- [ ] LLM-powered decision making
- [ ] Custom agent personalities

### Advanced Integrations
- [ ] Fine-tune LLM on financial simulation data
- [ ] Integrate with knowledge graph for deeper explanations
- [ ] Multi-agent conversations (agents talking to each other)
- [ ] Visual explanations (LLM generates charts)

## Credits
- **LLM Provider:** Featherless AI (https://featherless.ai)
- **Model:** Meta Llama 3.1 70B Instruct
- **UI Framework:** shadcn/ui + Radix UI
- **Icons:** Lucide React

## License
Part of RUDRA project - see root LICENSE file.
