# Maya — Malayalam AI Receptionist Prototype

24×7 voice AI receptionist demo powered by **Sarvam AI** (STT + TTS) and **Anthropic Claude** (LLM).
Handles Malayalam, English, and natural code-switching.

---

## Quick Start

```bash
# 1. Enter the project folder
cd ai-receptionist

# 2. First run — creates .env from template
bash run.sh

# 3. Edit .env and fill in your API keys
#    SARVAM_API_KEY    → https://dashboard.sarvam.ai
#    ANTHROPIC_API_KEY → https://console.anthropic.com

# 4. Run again to start the server
bash run.sh

# 5. Open browser
open http://localhost:8000
```

> Works without API keys — backend returns mock responses so the UI is fully explorable.

---

## Architecture

```
Browser (WAV @ 16kHz)
     │
     ▼  WebSocket /ws/{session_id}
FastAPI Backend (main.py)
     │
     ├─ Sarvam STT  ──► transcript
     ├─ Claude LLM  ──► Malayalam response
     └─ Sarvam TTS  ──► audio bytes
     │
     ▼  WebSocket events
Browser (plays audio + shows event log)
```

### Pipeline latency targets (India-hosted)

| Stage  | Target   |
|--------|----------|
| STT    | 200–400ms |
| LLM    | 300–700ms |
| TTS    | 80–150ms  |
| Total  | < 1.3s    |

---

## Project Structure

```
ai-receptionist/
├── main.py           — FastAPI + WebSocket server
├── requirements.txt  — Python dependencies
├── .env.example      — Config template
├── run.sh            — One-command startup
└── frontend/
    └── index.html    — Complete single-file UI
```

---

## Customising for Your Client

Edit `.env`:

```env
BUSINESS_NAME=Your Business Name
BUSINESS_HOURS=Monday to Friday, 9 AM to 5 PM
BUSINESS_SERVICES=sales, support, and booking
```

Maya's persona and knowledge are driven purely by these values — no code change needed.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service status + config check |
| `GET /api/events` | All logged events (optional `?session_id=`) |
| `GET /api/sessions` | Active WebSocket sessions |
| `WS  /ws/{session_id}` | Real-time voice pipeline |

---

## Sarvam AI Models Used

| Task | Model |
|------|-------|
| Speech-to-Text | `saarika:v1` |
| Text-to-Speech | `bulbul:v1` (voice: meera) |

---

## Production Notes

- Swap in-memory `event_store` for Redis or Postgres for persistence
- Add auth (JWT or API key) to the `/ws` endpoint before client-facing deployment
- For telephony: connect Exotel/Tata Comm SIP → this WebSocket via a SIP↔WS bridge
- Host on AWS Mumbai (`ap-south-1`) for lowest latency to Kerala + GCC users
