"""
Maya — AI Receptionist Backend
================================
FastAPI + WebSocket server powering the Maya Malayalam AI receptionist demo.

Pipeline per voice turn:
  Browser WAV → Sarvam STT → Claude LLM → Sarvam TTS → Browser Audio

All events are logged to file + console + pushed live to the frontend via WebSocket.
"""

import asyncio
import base64
import json
import logging
import math
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Logging — writes to both console and maya.log
# ─────────────────────────────────────────────────────────────────
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh = logging.FileHandler("maya.log", encoding="utf-8")
_fh.setFormatter(fmt)
_ch = logging.StreamHandler()
_ch.setFormatter(fmt)

logger = logging.getLogger("maya")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.addHandler(_ch)

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
SARVAM_API_KEY    = os.getenv("SARVAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BUSINESS_NAME     = os.getenv("BUSINESS_NAME", "Demo Business")
BUSINESS_HOURS    = os.getenv("BUSINESS_HOURS", "Monday to Saturday, 9 AM to 6 PM")
BUSINESS_SERVICES = os.getenv("BUSINESS_SERVICES", "consulting, support, and appointments")

_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ─────────────────────────────────────────────────────────────────
# Knowledge Base — loaded from kb_index.json (built by ingest.py)
# ─────────────────────────────────────────────────────────────────
KB_INDEX_PATH = Path("kb_index.json")
KB_CHUNKS: list[dict] = []          # list of {id, source, text}
_KB_IDF:  dict[str, float] = {}     # pre-computed IDF weights


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zഀ-ൿ]{2,}", text.lower())


def _load_kb() -> None:
    """Load and index kb_index.json.  Called once at startup."""
    global KB_CHUNKS, _KB_IDF

    if not KB_INDEX_PATH.exists():
        logger.info("No kb_index.json found — run `python ingest.py` to build a knowledge base.")
        return

    data       = json.loads(KB_INDEX_PATH.read_text(encoding="utf-8"))
    KB_CHUNKS  = data.get("chunks", [])
    n_docs     = len(KB_CHUNKS)

    if n_docs == 0:
        logger.info("kb_index.json is empty.")
        return

    # Compute IDF (inverse document frequency) for BM25-style scoring
    df: dict[str, int] = {}
    for chunk in KB_CHUNKS:
        for tok in set(_tokenize(chunk["text"])):
            df[tok] = df.get(tok, 0) + 1

    _KB_IDF = {tok: math.log((n_docs + 1) / (freq + 1)) + 1.0
               for tok, freq in df.items()}

    logger.info(
        "Knowledge base loaded: %d chunks from %d source(s) (built %s)",
        n_docs,
        len({c["source"] for c in KB_CHUNKS}),
        data.get("built_at", "unknown"),
    )


def search_kb(query: str, top_k: int = 3, min_score: float = 0.05) -> list[str]:
    """
    BM25-lite retrieval — returns the top_k most relevant chunks as plain strings.
    Returns an empty list when the KB is empty or nothing is relevant.
    """
    if not KB_CHUNKS:
        return []

    k1, b = 1.5, 0.75
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    avg_dl  = sum(len(_tokenize(c["text"])) for c in KB_CHUNKS) / len(KB_CHUNKS)
    scored: list[tuple[float, str]] = []

    for chunk in KB_CHUNKS:
        doc_tokens = _tokenize(chunk["text"])
        dl         = len(doc_tokens)
        tf_map: dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for tok in q_tokens:
            if tok not in _KB_IDF:
                continue
            tf  = tf_map.get(tok, 0)
            idf = _KB_IDF[tok]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))

        if score >= min_score:
            scored.append((score, chunk["text"]))

    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

# ─────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Maya AI Receptionist", version="1.0.0")

# Load knowledge base once at server start
_load_kb()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (sufficient for demo; swap for Redis in production)
event_store: list[dict] = []
sessions: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────
# Event Logging
# ─────────────────────────────────────────────────────────────────
def log_event(session_id: str, event_type: str, data: dict | None = None) -> dict:
    """Persist an event to the in-memory store and log file."""
    event = {
        "id": str(uuid.uuid4())[:8],
        "session_id": session_id,
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data or {},
    }
    event_store.append(event)
    logger.info(
        "[%s] %-22s | %s",
        session_id[:8],
        event_type,
        json.dumps(data or {}, ensure_ascii=False),
    )
    return event


# ─────────────────────────────────────────────────────────────────
# Sarvam STT
# ─────────────────────────────────────────────────────────────────
async def sarvam_stt(audio_bytes: bytes, language: str = "ml-IN") -> dict:
    """
    Send audio to Sarvam Speech-to-Text.
    Accepts WAV (16 kHz mono) built in the browser.
    Falls back to a mock transcript when SARVAM_API_KEY is not set.
    """
    if not SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set — returning mock STT transcript")
        return {"transcript": "[Mock] ഹലോ, ഇത് ഒരു ടെസ്റ്റ് ആണ്", "language_code": language}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files=[("file", ("audio.wav", audio_bytes, "audio/wav"))],
            data={"language_code": language, "model": "saarika:v2.5"},
        )
        if not response.is_success:
            logger.error("Sarvam STT error %s: %s", response.status_code, response.text)
        response.raise_for_status()
        return response.json()


# ─────────────────────────────────────────────────────────────────
# Claude LLM
# ─────────────────────────────────────────────────────────────────
MAYA_SYSTEM = """You are Maya, a warm and professional AI receptionist for {business}, a security electronics company in Cochin, Kerala.

Business details:
- Office hours: {hours} IST
- Contact: +91 9656 069 703 / +91 9747 554 459
- Location: Ernakulam, Cochin

Core products and services (use the KNOWLEDGE BASE below for full details):
{services}

Voice response rules (STRICT):
1. Respond in Malayalam by default. Handle Malayalam-English code-switching naturally.
2. ALWAYS answer from the KNOWLEDGE BASE when it contains relevant information — never invent products, prices, or details not mentioned there.
3. Maximum 2-3 sentences — this is spoken voice, not text.
4. Be helpful, warm, and professional.
5. When asked about specific products or pricing, give the actual details from the knowledge base — do NOT give a generic list.
6. For anything you cannot answer, politely offer to connect the caller to a human agent or suggest calling +91 9656 069 703.
7. Write Malayalam in proper Unicode script (not transliteration).
8. No bullet points, no lists — speak naturally as if on a phone call.
9. NEVER start a response with filler phrases like "ഞാൻ ഉടൻ പറയാം", "ഞാൻ പറയാം", "ശരി പറയാം", "ഞാൻ അറിയിക്കാം", or any variation of "I will tell you". Go straight to the answer — a real person on a call does not announce they are about to speak, they just speak."""


# Filler phrases Maya says while thinking (played over the silence gap)
FILLERS = [
    "ഹ്മ്മ്...",
    "ഓ...",
    "അതെ...",
]


async def llm_respond(
    transcript: str,
    history: list[dict],
    language: str = "ml-IN",
    kb_chunks: list[str] | None = None,
) -> str:
    if not _anthropic:
        return "[Mock] ഞങ്ങൾ തിങ്കൾ മുതൽ ശനി വരെ, രാവിലെ 9 മുതൽ വൈകിട്ട് 6 വരെ തുറന്നിരിക്കും. ANTHROPIC_API_KEY കോൺഫിഗർ ചെയ്യുക."

    system = MAYA_SYSTEM.format(
        business=BUSINESS_NAME,
        hours=BUSINESS_HOURS,
        services=BUSINESS_SERVICES,
    )

    # Inject retrieved knowledge base chunks when available
    if kb_chunks:
        context_block = "\n\n---\n\n".join(kb_chunks)
        system += (
            "\n\n--- KNOWLEDGE BASE (use this to answer accurately) ---\n"
            + context_block
            + "\n--- END KNOWLEDGE BASE ---"
            "\n\nIMPORTANT: If the answer is in the knowledge base, use it precisely. "
            "Never invent prices, hours, or details not mentioned above."
        )

    # Keep last 12 turns to stay within context
    messages = (history[-12:]) + [{"role": "user", "content": transcript}]

    resp = _anthropic.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


# ─────────────────────────────────────────────────────────────────
# Sarvam TTS
# ─────────────────────────────────────────────────────────────────
async def sarvam_tts(text: str, language: str = "ml-IN") -> bytes:
    """
    Convert text to speech via Sarvam Bulbul TTS.
    Returns raw WAV bytes. Returns empty bytes when key is absent.
    """
    if not SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set — skipping TTS audio")
        return b""

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "inputs": [text],
                "target_language_code": language,
                "speaker": "anushka",
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
        )
        if not response.is_success:
            logger.error("Sarvam TTS error %s: %s", response.status_code, response.text)
        response.raise_for_status()
        result = response.json()
        return base64.b64decode(result["audios"][0])


def _detect_tts_language(text: str) -> str:
    """Use ml-IN if the response contains Malayalam Unicode chars, else en-IN."""
    return "ml-IN" if any(0x0D00 <= ord(c) <= 0x0D7F for c in text) else "en-IN"


# ─────────────────────────────────────────────────────────────────
# WebSocket helpers
# ─────────────────────────────────────────────────────────────────
async def ws_send(ws: WebSocket, msg_type: str, payload: dict, session_id: str = "") -> None:
    """Send a typed JSON message to the connected browser client."""
    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": msg_type,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    **payload,
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        logger.warning("ws_send failed: %s", exc)


# ─────────────────────────────────────────────────────────────────
# WebSocket Endpoint — main session handler
# ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def ws_session(ws: WebSocket, session_id: str):
    await ws.accept()

    session: dict = {
        "id": session_id,
        "history": [],          # Claude conversation history
        "language": "ml-IN",    # Active STT/TTS language
        "start_time": time.time(),
        "turns": 0,
    }
    sessions[session_id] = session
    log_event(session_id, "session_started", {"language": "ml-IN"})

    # ── Greeting ────────────────────────────────────────────────
    try:
        await ws_send(ws, "status", {"status": "initializing"}, session_id)

        greeting = "നമസ്കാരം! SGI Netronics-ലേക്ക് സ്വാഗതം. ഞാൻ Maya. എങ്ങനെ സഹായിക്കണം?"

        t0 = time.time()
        audio = await sarvam_tts(greeting, "ml-IN")
        tts_ms = round((time.time() - t0) * 1000)

        log_event(session_id, "greeting_sent", {"text": greeting, "tts_ms": tts_ms})
        await ws_send(
            ws,
            "greeting",
            {
                "text": greeting,
                "audio": base64.b64encode(audio).decode() if audio else "",
                "tts_ms": tts_ms,
            },
            session_id,
        )
        session["history"].append({"role": "assistant", "content": greeting})
        await ws_send(ws, "status", {"status": "listening"}, session_id)

    except Exception as exc:
        logger.error("Greeting failed: %s", exc, exc_info=True)
        log_event(session_id, "error", {"phase": "greeting", "error": str(exc)})
        await ws_send(ws, "status", {"status": "listening"}, session_id)

    # ── Main receive loop ────────────────────────────────────────
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            match msg.get("type"):
                case "audio":
                    await _process_turn(ws, session, msg)
                case "set_language":
                    session["language"] = msg.get("language", "ml-IN")
                    log_event(session_id, "language_changed", {"language": session["language"]})
                case "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        dur = round(time.time() - session["start_time"])
        log_event(session_id, "session_ended", {"duration_s": dur, "turns": session["turns"]})
        sessions.pop(session_id, None)
    except Exception as exc:
        logger.error("WS loop error [%s]: %s", session_id, exc, exc_info=True)
        log_event(session_id, "error", {"error": str(exc)})
        sessions.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────
# Filler — played over the silence gap while LLM runs
# ─────────────────────────────────────────────────────────────────
async def _send_filler(ws: WebSocket, sid: str, delay: float = 0.9) -> None:
    """Generate and push a filler phrase only if the LLM is taking a while.
    The delay ensures fast responses (< 900 ms) never get a filler at all."""
    try:
        await asyncio.sleep(delay)
        phrase = random.choice(FILLERS)
        audio = await sarvam_tts(phrase, "ml-IN")
        if audio:
            await ws_send(
                ws,
                "filler_audio",
                {"audio": base64.b64encode(audio).decode(), "text": phrase},
                sid,
            )
            log_event(sid, "filler_sent", {"text": phrase})
    except asyncio.CancelledError:
        pass   # LLM finished first — filler not needed, silently dropped
    except Exception as exc:
        logger.warning("Filler TTS failed: %s", exc)


# ─────────────────────────────────────────────────────────────────
# Core pipeline — one voice turn
# ─────────────────────────────────────────────────────────────────
async def _process_turn(ws: WebSocket, session: dict, msg: dict) -> None:
    sid = session["id"]
    t_turn = time.time()

    try:
        audio_bytes = base64.b64decode(msg["audio"])
        log_event(sid, "audio_received", {"bytes": len(audio_bytes)})

        # ── 1. Speech → Text ──────────────────────────────────
        await ws_send(ws, "status", {"status": "transcribing"}, sid)
        log_event(sid, "stt_started", {})
        t0 = time.time()

        stt_result = await sarvam_stt(audio_bytes, session["language"])
        transcript  = (stt_result.get("transcript") or "").strip()
        stt_ms      = round((time.time() - t0) * 1000)

        log_event(
            sid,
            "stt_completed",
            {
                "transcript": transcript,
                "latency_ms": stt_ms,
                "detected_lang": stt_result.get("language_code", ""),
            },
        )

        if not transcript:
            await ws_send(ws, "notice", {"message": "Audio unclear — please try again."}, sid)
            await ws_send(ws, "status", {"status": "listening"}, sid)
            return

        await ws_send(ws, "transcript", {"text": transcript, "latency_ms": stt_ms}, sid)

        # ── 2. KB retrieval + LLM + Filler (concurrent) ───────
        # KB search is fast (in-memory BM25); filler TTS fires concurrently
        # with the LLM call so there's no dead air while Claude thinks.
        await ws_send(ws, "status", {"status": "thinking"}, sid)

        kb_chunks = search_kb(transcript)
        if kb_chunks:
            log_event(sid, "kb_retrieved", {
                "query": transcript,
                "chunks_found": len(kb_chunks),
                "preview": kb_chunks[0][:80] + "…" if kb_chunks else "",
            })

        log_event(sid, "llm_started", {"prompt": transcript})
        t1 = time.time()

        response_text = await llm_respond(
            transcript, session["history"], session["language"], kb_chunks
        )
        llm_ms = round((time.time() - t1) * 1000)

        log_event(sid, "llm_completed", {"response": response_text, "latency_ms": llm_ms})
        await ws_send(ws, "response_text", {"text": response_text, "latency_ms": llm_ms}, sid)

        session["history"].extend(
            [
                {"role": "user",      "content": transcript},
                {"role": "assistant", "content": response_text},
            ]
        )
        session["turns"] += 1

        # ── 3. Text → Speech ──────────────────────────────────
        await ws_send(ws, "status", {"status": "speaking"}, sid)
        log_event(sid, "tts_started", {})
        t2 = time.time()

        tts_lang  = _detect_tts_language(response_text)
        audio_out = await sarvam_tts(response_text, tts_lang)
        tts_ms    = round((time.time() - t2) * 1000)
        total_ms  = round((time.time() - t_turn) * 1000)

        log_event(
            sid,
            "tts_completed",
            {"latency_ms": tts_ms, "audio_bytes": len(audio_out), "total_turn_ms": total_ms},
        )

        await ws_send(
            ws,
            "audio_response",
            {
                "audio": base64.b64encode(audio_out).decode() if audio_out else "",
                "text": response_text,
                "metrics": {
                    "stt_ms":   stt_ms,
                    "llm_ms":   llm_ms,
                    "tts_ms":   tts_ms,
                    "total_ms": total_ms,
                },
            },
            sid,
        )
        # Tell browser audio is queued — it will reactivate VAD after playback
        await ws_send(ws, "status", {"status": "listening"}, sid)

    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400]
        err = f"API {exc.response.status_code}: {body}"
        logger.error("HTTP error in turn [%s]: %s", sid, err)
        log_event(sid, "error", {"error": err})
        await ws_send(ws, "error", {"message": err}, sid)
        await ws_send(ws, "status", {"status": "listening"}, sid)
    except Exception as exc:
        logger.error("Turn error [%s]: %s", sid, exc, exc_info=True)
        log_event(sid, "error", {"error": str(exc)})
        await ws_send(ws, "error", {"message": str(exc)}, sid)
        await ws_send(ws, "status", {"status": "listening"}, sid)


# ─────────────────────────────────────────────────────────────────
# REST — diagnostics & event export
# ─────────────────────────────────────────────────────────────────
@app.get("/api/events")
async def get_events(session_id: Optional[str] = None, limit: int = 200):
    """Return logged events, optionally filtered by session_id."""
    events = event_store if not session_id else [
        e for e in event_store if e["session_id"] == session_id
    ]
    return events[-limit:]


@app.get("/api/sessions")
async def get_sessions():
    return {
        "active": len(sessions),
        "sessions": [
            {
                "id":         s["id"],
                "turns":      s["turns"],
                "duration_s": round(time.time() - s["start_time"]),
                "language":   s["language"],
            }
            for s in sessions.values()
        ],
    }


@app.post("/api/kb/reload")
async def reload_kb():
    """Hot-reload the knowledge base without restarting the server."""
    _load_kb()
    return {
        "status": "reloaded",
        "chunks": len(KB_CHUNKS),
        "sources": list({c["source"] for c in KB_CHUNKS}),
    }


@app.get("/api/kb/search")
async def kb_search(q: str, k: int = 3):
    """Debug endpoint — test KB retrieval for a given query."""
    chunks = search_kb(q, top_k=k)
    return {"query": q, "chunks_found": len(chunks), "results": chunks}


@app.get("/health")
async def health():
    return {
        "status":               "ok",
        "sarvam_configured":    bool(SARVAM_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "active_sessions":      len(sessions),
        "total_events":         len(event_store),
        "kb_chunks":            len(KB_CHUNKS),
        "kb_sources":           list({c["source"] for c in KB_CHUNKS}),
    }


# ─────────────────────────────────────────────────────────────────
# Static frontend — must be LAST (catch-all)
# ─────────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
