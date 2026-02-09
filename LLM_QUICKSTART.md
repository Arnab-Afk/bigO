# Quick Start: LLM Chatbot Setup

## 1. Get Featherless AI API Key

### Sign up (Free Tier Available)
```bash
# Visit: https://featherless.ai
# Create account
# Navigate to API Keys section
# Generate new API key
```

## 2. Configure Backend

### Set Environment Variable
```bash
cd backend

# Option 1: Add to .env file
echo "FEATHERLESS_API_KEY=your_actual_key_here" >> .env

# Option 2: Export in terminal (temporary)
export FEATHERLESS_API_KEY="your_actual_key_here"
```

### Verify Setup
```bash
# Start backend
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 17170 --reload

# In another terminal, test health endpoint
curl http://localhost:17170/api/v1/llm/health
```

**Expected Response:**
```json
{
  "status": "configured",
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "message": "LLM service ready"
}
```

**If Not Configured:**
```json
{
  "status": "not_configured",
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "message": "API key not set (set FEATHERLESS_API_KEY env var)"
}
```

## 3. Start Frontend

```bash
cd frontend
npm run dev
```

Visit: http://localhost:3000/ml-dashboard

## 4. Use the Chatbot

### Initialize Simulation
1. Complete entity onboarding (name your entity)
2. Click "Initialize Simulation"
3. Wait for network to load

### Open Chatbot
- Click the floating **message icon** (bottom-right corner)
- Chat window opens

### Automatic Narrations
As simulation runs, you'll see:
- 🤖 System messages with AI explanations
- Auto-updates when entities fail
- Explanations of shocks and decisions

### Ask Questions
Type in the input box:
- "What is happening right now?"
- "Why did that bank fail?"
- "Should I increase capital requirements?"

Press **Enter** to send (Shift+Enter for new line)

### Quick Actions
Use the button shortcuts:
- **System Health** - Get current status
- **Last Event** - Explain what just happened
- **Suggestions** - Get AI recommendations

## 5. Test Chat Manually

```bash
# Test a simple question
curl -X POST http://localhost:17170/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "test",
    "message": "Explain what system health means",
    "history": []
  }'
```

## 6. Troubleshooting

### No Narrations Appearing?

**Check 1: API Key Set?**
```bash
curl http://localhost:17170/api/v1/llm/health
```

**Check 2: WebSocket Connected?**
- Look for green "Live" badge in dashboard
- Check browser console for WebSocket errors

**Check 3: Backend Logs**
```bash
# Should see logs like:
# narration_generated simulation_id=... event_type=...
```

**Check 4: Browser Console**
```javascript
// Open DevTools (F12)
// Look for:
// 🤖 AI Narration: [explanation text]
```

### Slow Responses?
- LLM inference takes 2-10 seconds (normal)
- Typing indicator appears while waiting
- Consider using faster model:
  ```bash
  export FEATHERLESS_MODEL="meta-llama/Llama-3.2-8B-Instruct"
  ```

### API Rate Limit Hit?
- Free tier has limits
- Narrations are throttled automatically
- Check Featherless dashboard for usage

### CORS Errors?
```python
# backend/app/main.py already configured for:
allow_origins=["http://localhost:3000"]
```

## 7. Optional: Disable Auto-Narration

If you want chat-only (no auto-explanations):

```python
# backend/app/api/v1/abm_simulation.py
# Comment out narrate_event_background() calls

# Lines ~387, ~404, ~443, ~726
# if narrate_event_background:
#     narrate_event_background(...)
```

## 8. Features to Try

### During Simulation
- **Step through** - Watch narrations explain each timestep
- **Apply shocks** - Get AI explanation of impact
- **Make decisions** - Ask AI for advice before choosing

### Chatbot Questions
```
"What entities are most at risk?"
"Explain how contagion spreads"
"What does the capital ratio mean?"
"How can I prevent bank failures?"
"Compare this timestep to the previous one"
```

### Quick Experiments
- Run simulation with different policies
- Ask AI to compare outcomes
- Use chat as learning tool for financial networks

## 9. Next Steps

See [LLM_CHATBOT_COMPLETE.md](./LLM_CHATBOT_COMPLETE.md) for:
- Full architecture details
- API reference
- Advanced configuration
- Custom prompts
- Performance tuning

## 10. Demo Script

```bash
# Full demo setup (copy-paste friendly)

# Terminal 1: Backend
cd backend
source .venv/bin/activate
export FEATHERLESS_API_KEY="your_key"
python -m uvicorn app.main:app --host 0.0.0.0 --port 17170 --reload

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Test
curl http://localhost:17170/api/v1/llm/health
curl -X POST http://localhost:17170/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"simulation_id":"test","message":"Hello!","history":[]}'
```

## Need Help?
- Check backend logs for errors
- Inspect browser console (F12)
- Verify API key with health endpoint
- Review [LLM_CHATBOT_COMPLETE.md](./LLM_CHATBOT_COMPLETE.md)
