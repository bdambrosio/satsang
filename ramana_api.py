#!/usr/bin/env python3
"""
Ramana Website API Server
==========================

Flask API server for the Ramana Maharshi website.
Provides endpoints for:
- Random Nan_Yar passage
- Progressive query flow: response -> expand -> passages (non-blocking)
- Session tracking with conversation history
- Sidebar RAG retrieval with LLM post-filter
"""

import argparse
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from flask import Flask, jsonify, request, render_template, make_response

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aliveness_critic import ContemplativeGenerator, LocalCritic
from contemplative_rag import ContemplativeRAGProvider
from filtered_passages_rag import FilteredPassagesRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Global state (initialized on startup) ─────────────────────────────

nan_yar_passages = []
nan_yar_embeddings = None  # Pre-computed embeddings for personalized selection
nan_yar_embedder = None    # SentenceTransformer instance (shared with RAG)
generator = None
filtered_passages_rag = None
llm_http = httpx.Client(timeout=60.0)
llm_model_name = None

# ── LLM Configuration (set from CLI/env) ──────────────────────────────

llm_backend = "local"  # "local" or "openrouter"
llm_url = "http://localhost:5000/v1"  # vLLM URL or OpenRouter URL
llm_model = None  # Model name (required for OpenRouter, auto-detected for local)

# ── Constants ─────────────────────────────────────────────────────────

SESSIONS_DIR = Path(__file__).parent / "sessions"


# ── JSON Repair Utility ──────────────────────────────────────────────

def repair_json_response(raw: str, response_obj: dict = None) -> tuple[str, bool]:
    """
    Attempt to repair a malformed JSON response from an LLM.
    
    Steps:
    1. Remove markdown code fences
    2. Extract text between first '{'/'[' and last matching '}'/']'
    3. Check if response was truncated (via finish_reason or abrupt ending)
    
    Args:
        raw: Raw LLM response text
        response_obj: Full API response object (to check finish_reason)
    
    Returns:
        (repaired_json_string, was_truncated)
    """
    was_truncated = False
    
    # Check for truncation in response object
    if response_obj:
        choice = response_obj.get("choices", [{}])[0] if response_obj.get("choices") else {}
        finish_reason = choice.get("finish_reason", "")
        if finish_reason in ("length", "max_tokens"):
            was_truncated = True
            logger.warning(f"LLM response truncated (finish_reason={finish_reason})")
    
    # Remove markdown code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        # Remove closing fence
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()
    
    # Check if empty after cleaning
    if not cleaned:
        logger.warning("LLM returned empty response after cleaning")
        return "", was_truncated
    
    # Extract JSON between first '{'/'[' and last matching '}'/']'
    first_brace = cleaned.find('{')
    first_bracket = cleaned.find('[')
    
    if first_brace == -1 and first_bracket == -1:
        logger.warning(f"No JSON structure found in response. Raw: {raw[:200]}")
        return cleaned, was_truncated
    
    # Determine which comes first
    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        # JSON object
        start_char = '{'
        end_char = '}'
        start_idx = first_brace
    else:
        # JSON array
        start_char = '['
        end_char = ']'
        start_idx = first_bracket
    
    # Find matching closing brace/bracket
    depth = 0
    end_idx = -1
    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        # Unmatched brackets - likely truncated
        was_truncated = True
        logger.warning("Unmatched JSON brackets detected - response likely truncated")
        # Try to extract what we have
        extracted = cleaned[start_idx:]
    else:
        extracted = cleaned[start_idx:end_idx]
    
    # Remove any leading/trailing non-JSON text
    extracted = extracted.strip()
    
    return extracted, was_truncated

# Session cookie settings
SESSION_COOKIE_NAME = "ramana_session"
SESSION_COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 year

# Conversational mode system prompt
CONVERSATIONAL_SYSTEM = """You are a contemplative companion grounded in \
Ramana Maharshi's teachings. You speak warmly but without flattery. You \
invite inquiry rather than giving answers. You are patient with \
confusion and meet the questioner where they are, gently turning \
attention inward. You never prescribe practices or claim authority. \
When silence would serve, you say so honestly rather than filling space. \
You draw from Bhagavan's words and the spirit of his interactions with \
visitors as recorded in the Commentaries and Talks."""

# Session timeout for rollup (30 minutes of inactivity)
SESSION_TIMEOUT_SECONDS = 30 * 60

# Passage post-filter prompt
PASSAGE_FILTER_PROMPT = """You are curating a sidebar passage for a website devoted to Ramana Maharshi's teachings. A user asked a question and received a response. You must judge whether a candidate passage from another contemplative tradition would deepen the reader's understanding of the response, or risk confusing, misleading, or contradicting them.

The passage should:
- Illuminate the SAME experiential territory as the response, from a different angle
- Be coherent with the response's meaning (not contradict or undermine it)
- Be accessible enough that the reader can connect it to the dialogue
- Add genuine contemplative depth, not merely repeat the same idea in different words

The passage should NOT:
- Contradict the core pointing of the response
- Introduce confusing terminology or concepts that would distract
- Be so tangential that the connection requires specialist knowledge
- Present a fundamentally different path in a way that muddies the response

DIALOGUE:
Question: {user_input}
{user_summary_section}
Response: {response}

Expanded: {expanded_response}

--- CANDIDATE PASSAGE ---
{passage_text}
--- {passage_author}, {passage_title} ({passage_tradition}) ---

Respond with ONLY a JSON object. No markdown, no preamble, no introductory text, no explanation, no reasoning. Just the JSON:
{{
  "show": true or false,
  "confidence": <float 0.0 to 1.0>,
  "reason": "<terse, 6-10 words max>"
}}"""

# Expand post-filter prompt
EXPAND_FILTER_PROMPT = """You are reviewing an expanded response for a website devoted to Ramana Maharshi's teachings. The original response was a brief, direct pointing. The expanded version elaborates on it, drawing from Commentaries and related teachings.

You must judge whether the expanded response:
- FAITHFULLY elaborates on the original response without contradicting or distorting it
- Remains COHERENT and accessible to the reader
- Does NOT introduce confusing, misleading, or contradictory material
- Adds genuine depth rather than diluting the original pointing

DIALOGUE:
Question: {user_input}
{user_context_section}
Original Response: {response}

--- EXPANDED RESPONSE ---
{expanded_response}
--- END ---

If the expanded response fails these criteria, provide a brief rewrite instruction explaining what's wrong and how to fix it.

Respond with ONLY a JSON object. No markdown, no preamble, no introductory text, no explanation, no reasoning. Just the JSON:
{{
  "acceptable": true or false,
  "confidence": <float 0.0 to 1.0>,
  "reason": "<terse, 6-10 words max>",
  "rewrite_instruction": "<brief instruction if not acceptable, else null>"
}}"""

# Session rollup prompt: generates a user model from conversation history
SESSION_ROLLUP_PROMPT = """You are maintaining a contemplative interest profile for a visitor to a website devoted to Ramana Maharshi's teachings. Given this visitor's conversation history (and optionally their previous profile), produce an updated set of semantic threads that capture the contemplative territory this person is drawn to.

Each thread should be a brief description (under 30 words) of an experiential or contemplative interest, written in clear, tradition-neutral language. Focus on what territory of awareness, self, mind, realization, or lived contemplative experience they are exploring — NOT surface topics or conversation mechanics.

GOOD thread: "Inquiry into whether effort itself is an obstacle — the paradox of trying to stop trying."
BAD thread: "The user frequently asks about meditation techniques."

Produce 3-7 threads that form a coherent portrait of this visitor's contemplative engagement. Threads should evolve: if earlier interests have faded in recent conversations, let them go. If new territory has emerged, include it.

{previous_profile_section}
CONVERSATION HISTORY:
{conversation_history}

Respond with ONLY a JSON object. No markdown, no preamble, no introductory text, no explanation, no reasoning. Just the JSON:
{{
  "semantic_threads": ["<thread1>", "<thread2>", ...],
  "themes": ["<theme1>", "<theme2>", ...]
}}

themes must use the controlled vocabulary: self-inquiry, ego-dissolution, true-nature, witness-awareness, surrender, effortless-effort, beyond-knowledge, direct-experience, nature-of-mind, stillness, attention, detachment, illusion, unity, ordinary-life, impermanence, suffering, desire, liberation, death, guru-grace, devotion, compassion, the-source, presence, the-sacred, practice-as-life, paradox, body, scripture-and-tradition

No markdown, no preamble."""


# ── Session Management ────────────────────────────────────────────────

def get_or_create_session_id() -> str:
    """Get session_id from cookie or create a new one."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        return session_id
    return str(uuid.uuid4())


def get_session_file(session_id: str) -> Path:
    """Get path to session history file."""
    return SESSIONS_DIR / f"{session_id}.jsonl"


def get_rollup_file(session_id: str) -> Path:
    """Get path to session rollup file."""
    return SESSIONS_DIR / f"{session_id}.rollup.json"


def append_turn(session_id: str, turn: dict):
    """Append a conversation turn to the session file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_file = get_session_file(session_id)
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")


def load_session_history(session_id: str) -> list[dict]:
    """Load all turns from a session file."""
    session_file = get_session_file(session_id)
    if not session_file.exists():
        return []
    turns = []
    with open(session_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return turns


def load_user_model(session_id: str) -> dict:
    """
    Load the user model (rollup) for a session.
    
    Returns:
        {"semantic_threads": [...], "themes": [...], "shown_nan_yar_indices": [...], 
         "shown_sidebar_passage_ids": [...]} or empty dict
    """
    rollup_file = get_rollup_file(session_id)
    if not rollup_file.exists():
        return {}
    try:
        with open(rollup_file, "r") as f:
            return json.loads(f.read())
    except (json.JSONDecodeError, IOError):
        return {}


def get_shown_nan_yar_indices(session_id: str) -> set[int]:
    """Get set of Nan_Yar passage indices already shown to this session."""
    model = load_user_model(session_id)
    return set(model.get("shown_nan_yar_indices", []))


def mark_nan_yar_shown(session_id: str, passage_idx: int):
    """Mark a Nan_Yar passage as shown for this session."""
    rollup_file = get_rollup_file(session_id)
    model = load_user_model(session_id)
    
    shown = set(model.get("shown_nan_yar_indices", []))
    shown.add(passage_idx)
    model["shown_nan_yar_indices"] = list(shown)
    
    # Ensure rollup file directory exists
    rollup_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rollup_file, "w") as f:
        f.write(json.dumps(model))


def get_shown_sidebar_passage_ids(session_id: str) -> set[str]:
    """Get set of sidebar passage_ids already shown to this session."""
    model = load_user_model(session_id)
    return set(model.get("shown_sidebar_passage_ids", []))


def mark_sidebar_passage_shown(session_id: str, passage_id: str):
    """Mark a sidebar passage as shown for this session."""
    rollup_file = get_rollup_file(session_id)
    model = load_user_model(session_id)
    
    shown = set(model.get("shown_sidebar_passage_ids", []))
    shown.add(passage_id)
    model["shown_sidebar_passage_ids"] = list(shown)
    
    # Ensure rollup file directory exists
    rollup_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rollup_file, "w") as f:
        f.write(json.dumps(model))


def get_user_model_threads(session_id: str) -> list[str]:
    """Get the semantic threads from the user model, or empty list."""
    model = load_user_model(session_id)
    return model.get("semantic_threads", [])


def get_recent_turns_text(session_id: str, n: int = 3) -> str:
    """
    Format the last N conversation turns as a compact snippet.
    
    Returns:
        String like "Q: ...\nR: ...\nQ: ...\nR: ..." or ""
    """
    history = load_session_history(session_id)
    if not history:
        return ""
    
    recent = history[-n:]
    lines = []
    for turn in recent:
        q = turn.get("user_input", "")
        r = turn.get("response", "")
        if q:
            lines.append(f"Q: {q}")
        if r:
            lines.append(f"R: {r}")
    return "\n".join(lines)


def get_user_context_for_prompt(session_id: str) -> str:
    """
    Build a user context string for injection into prompts.
    Combines user model threads + recent turns.
    """
    threads = get_user_model_threads(session_id)
    recent = get_recent_turns_text(session_id, n=3)
    
    parts = []
    if threads:
        parts.append("Visitor's contemplative interests: " + "; ".join(threads))
    if recent:
        parts.append("Recent conversation:\n" + recent)
    
    if not parts:
        return ""
    return "\n\n".join(parts)


def check_and_rollup_stale_session(session_id: str):
    """
    Check if the session has timed out and needs a rollup.
    
    Called on each request. If the last turn is older than
    SESSION_TIMEOUT_SECONDS, generates a rollup.
    
    NOTE: This runs synchronously in the request path. The rollup makes
    an LLM call which can take several seconds. A user returning after
    30 minutes of inactivity will experience a slow first response while
    the rollup generates. Consider backgrounding this in production.
    """
    history = load_session_history(session_id)
    if not history:
        return
    
    # Check if rollup already exists and is newer than last turn
    rollup_file = get_rollup_file(session_id)
    if rollup_file.exists():
        try:
            with open(rollup_file, "r") as f:
                rollup = json.loads(f.read())
            rollup_turn_count = rollup.get("turn_count", 0)
            # Only re-rollup if there are new turns since last rollup
            if rollup_turn_count >= len(history):
                return
        except (json.JSONDecodeError, IOError):
            pass
    
    # Check if last turn is stale
    last_turn = history[-1]
    last_ts = last_turn.get("timestamp", "")
    if not last_ts:
        return
    
    try:
        last_time = datetime.fromisoformat(last_ts)
        now = datetime.now(timezone.utc)
        elapsed = (now - last_time).total_seconds()
        
        if elapsed >= SESSION_TIMEOUT_SECONDS:
            logger.info(f"Session {session_id[:8]}... timed out ({elapsed:.0f}s), generating rollup")
            generate_session_rollup(session_id)
    except (ValueError, TypeError):
        pass


def generate_session_rollup(session_id: str):
    """
    Generate a session rollup: LLM-generated semantic threads summarizing
    the contemplative territory explored across all turns.
    
    Reads previous rollup (if any) so the model evolves rather than
    being rebuilt from scratch each time.
    """
    history = load_session_history(session_id)
    if not history:
        return
    
    # Build conversation history text
    conv_lines = []
    for turn in history:
        q = turn.get("user_input", "")
        r = turn.get("response", "")
        if q:
            conv_lines.append(f"Q: {q}")
        if r:
            conv_lines.append(f"R: {r}")
    conversation_text = "\n".join(conv_lines)
    
    # Check for previous rollup
    previous_profile_section = ""
    existing_model = load_user_model(session_id)
    prev_threads = existing_model.get("semantic_threads", [])
    if prev_threads:
        previous_profile_section = (
            "PREVIOUS PROFILE (update, don't start from scratch):\n"
            + "\n".join(f"- {t}" for t in prev_threads)
            + "\n\n"
        )
    
    prompt = SESSION_ROLLUP_PROMPT.format(
        previous_profile_section=previous_profile_section,
        conversation_history=conversation_text,
    )
    
    try:
        # Build request payload
        payload = {
            "model": llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        
        # Add reasoning parameter for OpenRouter (if supported)
        if llm_backend == "openrouter":
            payload["reasoning"] = {"effort": "low"}
        
        # Build headers (OpenRouter needs auth)
        headers = {}
        if llm_backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable required")
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "https://github.com/ramana-website"
            headers["X-Title"] = "Ramana Website"
        
        resp = llm_http.post(
            f"{llm_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        response_data = resp.json()
        raw = response_data["choices"][0]["message"]["content"].strip()
        
        # Repair JSON response
        repaired, was_truncated = repair_json_response(raw, response_data)
        if was_truncated:
            logger.warning("Session rollup response was truncated - JSON may be incomplete")
        if not repaired:
            raise ValueError("Empty JSON after repair")
        
        result = json.loads(repaired)
        
        # Preserve shown tracking fields from existing model
        existing_model = load_user_model(session_id)
        rollup = {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": len(history),
            "semantic_threads": result.get("semantic_threads", []),
            "themes": result.get("themes", []),
            "shown_nan_yar_indices": existing_model.get("shown_nan_yar_indices", []),
            "shown_sidebar_passage_ids": existing_model.get("shown_sidebar_passage_ids", []),
        }
        
        rollup_file = get_rollup_file(session_id)
        with open(rollup_file, "w") as f:
            f.write(json.dumps(rollup))
        
        logger.info(f"Session rollup generated for {session_id[:8]}... "
                    f"({len(history)} turns, {len(rollup['semantic_threads'])} threads)")
        for i, t in enumerate(rollup["semantic_threads"]):
            logger.info(f"  Thread {i+1}: {t}")
    
    except Exception as e:
        logger.error(f"Session rollup LLM call failed: {e}", exc_info=True)
        if 'raw' in locals():
            logger.error(f"Raw response (first 500 chars): {raw[:500]}")
        # Write a minimal rollup so we don't retry constantly
        # NOTE: If LLM fails and there was no prior rollup, prev_threads will be []
        # and we'll write an empty rollup with the current turn_count. This means
        # we won't retry the rollup for those turns. This is acceptable (avoids
        # infinite retry loops), but means rollup may be skipped for that session.
        rollup = {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": len(history),
            "semantic_threads": prev_threads,  # Keep previous if LLM fails
            "themes": existing_model.get("themes", []),
            "shown_nan_yar_indices": existing_model.get("shown_nan_yar_indices", []),
            "shown_sidebar_passage_ids": existing_model.get("shown_sidebar_passage_ids", []),
        }
        rollup_file = get_rollup_file(session_id)
        with open(rollup_file, "w") as f:
            f.write(json.dumps(rollup))


def set_session_cookie(resp, session_id: str):
    """Set the session cookie on a response."""
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
    )
    return resp


# ── LLM Passage Filter ───────────────────────────────────────────────

def filter_passage_with_llm(
    passage: dict,
    user_input: str,
    response: str,
    expanded_response: str,
    user_context: str = "",
) -> dict:
    """
    Ask LLM whether a candidate passage is coherent/aligned with the dialogue.
    
    Returns:
        {"show": bool, "confidence": float, "reason": str}
    """
    user_summary_section = ""
    if user_context:
        user_summary_section = f"Visitor context:\n{user_context}\n"
    
    prompt = PASSAGE_FILTER_PROMPT.format(
        user_input=user_input,
        user_summary_section=user_summary_section,
        response=response,
        expanded_response=expanded_response or "(none)",
        passage_text=passage.get("text", ""),
        passage_author=passage.get("author", "Unknown"),
        passage_title=passage.get("title", "Untitled"),
        passage_tradition=passage.get("tradition", "unknown"),
    )
    
    try:
        if llm_model_name == "unknown":
            logger.error("Cannot filter passage: llm_model_name is 'unknown'. Check LLM configuration.")
            return {"show": False, "confidence": 0.0, "reason": "LLM unavailable"}
        
        # Build request payload
        payload = {
            "model": llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
        }
        
        # Add reasoning parameter for OpenRouter (if supported)
        if llm_backend == "openrouter":
            payload["reasoning"] = {"effort": "low"}
        
        # Build headers (OpenRouter needs auth)
        headers = {}
        if llm_backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable required")
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "https://github.com/ramana-website"
            headers["X-Title"] = "Ramana Website"
        
        resp = llm_http.post(
            f"{llm_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        response_data = resp.json()
        raw = response_data["choices"][0]["message"]["content"].strip()
        
        # Repair JSON response
        repaired, was_truncated = repair_json_response(raw, response_data)
        if was_truncated:
            logger.warning("Passage filter response was truncated - JSON may be incomplete")
        if not repaired:
            raise ValueError("Empty JSON after repair")
        
        result = json.loads(repaired)
        
        return {
            "show": bool(result.get("show", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    
    except Exception as e:
        logger.error(f"Passage filter LLM call failed: {e}")
        if 'raw' in locals():
            logger.error(f"Raw response (first 500 chars): {raw[:500]}")
        # On failure, allow the passage through with low confidence
        return {"show": True, "confidence": 0.3, "reason": "filter unavailable"}


def filter_expand_with_llm(
    user_input: str,
    response: str,
    expanded_response: str,
    user_context: str = "",
) -> dict:
    """
    Ask LLM whether an expanded response is coherent/faithful to the original.
    
    Returns:
        {"acceptable": bool, "confidence": float, "reason": str,
         "rewrite_instruction": str or None}
    """
    user_context_section = ""
    if user_context:
        user_context_section = f"Visitor context:\n{user_context}\n"
    
    prompt = EXPAND_FILTER_PROMPT.format(
        user_input=user_input,
        user_context_section=user_context_section,
        response=response,
        expanded_response=expanded_response,
    )
    
    try:
        if llm_model_name == "unknown":
            logger.error("Cannot filter expand: llm_model_name is 'unknown'. Check LLM configuration.")
            return {"acceptable": False, "confidence": 0.0, "reason": "LLM unavailable",
                    "rewrite_instruction": None}
        
        payload = {
            "model": llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
        }
        
        # Add reasoning parameter for OpenRouter (if supported)
        if llm_backend == "openrouter":
            payload["reasoning"] = {"effort": "low"}
        
        headers = {}
        if llm_backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable required")
            headers["Authorization"] = f"Bearer {api_key}"
            headers["HTTP-Referer"] = "https://github.com/ramana-website"
            headers["X-Title"] = "Ramana Website"
        
        resp = llm_http.post(
            f"{llm_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        response_data = resp.json()
        raw = response_data["choices"][0]["message"]["content"].strip()
        
        # Repair JSON response
        repaired, was_truncated = repair_json_response(raw, response_data)
        if was_truncated:
            logger.warning("Expand filter response was truncated - JSON may be incomplete")
        if not repaired:
            raise ValueError("Empty JSON after repair")
        
        result = json.loads(repaired)
        
        return {
            "acceptable": bool(result.get("acceptable", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
            "rewrite_instruction": result.get("rewrite_instruction"),
        }
    
    except Exception as e:
        logger.error(f"Expand filter LLM call failed: {e}")
        if 'raw' in locals():
            logger.error(f"Raw response (first 500 chars): {raw[:500]}")
        # On failure, accept the expand
        return {"acceptable": True, "confidence": 0.3, "reason": "filter unavailable",
                "rewrite_instruction": None}


def detect_llm_model():
    """Auto-detect model from local vLLM API or use configured model."""
    global llm_model_name
    
    if llm_backend == "openrouter":
        # OpenRouter requires explicit model name
        if llm_model:
            llm_model_name = llm_model
            logger.info(f"Using OpenRouter model: {llm_model_name}")
        else:
            # Default for OpenRouter
            llm_model_name = "anthropic/claude-sonnet-4"
            logger.info(f"Using default OpenRouter model: {llm_model_name}")
    else:
        # Local vLLM: try to auto-detect
        try:
            resp = llm_http.get(f"{llm_url}/models")
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                llm_model_name = data["data"][0]["id"]
                logger.info(f"Auto-detected LLM model: {llm_model_name}")
            else:
                llm_model_name = llm_model or "unknown"
        except Exception as e:
            logger.warning(f"Could not detect LLM model: {e}")
            llm_model_name = llm_model or "unknown"


# ── Initialization ────────────────────────────────────────────────────

def load_nan_yar_passages():
    """Load Ramana passages from nan-yar.txt, Ulladu_Narpadu.txt, and Upadesa_Undiyar.txt."""
    global nan_yar_passages, nan_yar_embeddings, nan_yar_embedder
    
    ramana_dir = Path(__file__).parent / "ramana"
    all_passages = []  # List of dicts: {"text": str, "source": str}
    
    # Load passages from each file
    source_files = [
        ("nan-yar.txt", "Nan Yar"),
        ("Ulladu_Narpadu.txt", "Ulladu Narpadu"),
        ("Upadesa_Undiyar.txt", "Upadesa Undiyar"),
    ]
    
    for filename, source_name in source_files:
        file_path = ramana_dir / filename
        if not file_path.exists():
            logger.warning(f"{source_name} file not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try splitting by double newline first, then single newline
            passage_texts = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not passage_texts:
                passage_texts = [p.strip() for p in content.split('\n') if p.strip()]
            
            # Store as dicts with source attribution
            for text in passage_texts:
                all_passages.append({"text": text, "source": source_name})
            
            logger.info(f"Loaded {len(passage_texts)} passages from {source_name}")
        except Exception as e:
            logger.error(f"Error loading {source_name}: {e}")
            continue
    
    if not all_passages:
        logger.warning("No Ramana passages loaded from any source")
        nan_yar_passages = []
        nan_yar_embeddings = None
        nan_yar_embedder = None
        return
    
    nan_yar_passages = all_passages
    logger.info(f"Loaded {len(nan_yar_passages)} total Ramana passages")
    
    # Pre-embed for personalized selection (use text field for embedding)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
        
        nan_yar_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        passage_texts = [p["text"] for p in nan_yar_passages]
        embeddings = nan_yar_embedder.encode(passage_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        nan_yar_embeddings = embeddings
        logger.info(f"Pre-embedded {len(nan_yar_passages)} Ramana passages for personalized selection")
    except Exception as e:
        logger.warning(f"Could not pre-embed Ramana passages (will use random): {e}")
        nan_yar_embeddings = None
        nan_yar_embedder = None


def select_nan_yar_passage(session_id: str) -> dict:
    """
    Select a Ramana passage (from Nan Yar, Ulladu Narpadu, or Upadesa Undiyar).
    Personalized if user model exists, otherwise random.
    Excludes passages already shown in this session.
    
    Returns:
        {"text": str, "source": str}
    """
    if not nan_yar_passages:
        return {"text": "", "source": ""}
    
    shown_indices = get_shown_nan_yar_indices(session_id)
    available_indices = [i for i in range(len(nan_yar_passages)) if i not in shown_indices]
    
    # If all passages have been shown, reset and start over
    if not available_indices:
        logger.info(f"All Ramana passages shown for {session_id[:8]}..., resetting")
        shown_indices.clear()
        available_indices = list(range(len(nan_yar_passages)))
        # Clear the shown list
        rollup_file = get_rollup_file(session_id)
        model = load_user_model(session_id)
        model["shown_nan_yar_indices"] = []
        rollup_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rollup_file, "w") as f:
            f.write(json.dumps(model))
    
    # Try personalized selection
    if nan_yar_embeddings is not None and nan_yar_embedder is not None:
        threads = get_user_model_threads(session_id)
        if threads:
            try:
                import numpy as np
                import faiss
                
                # Embed user model threads, take mean
                thread_vecs = nan_yar_embedder.encode(threads, show_progress_bar=False)
                thread_vecs = np.array(thread_vecs).astype('float32')
                query_vec = np.mean(thread_vecs, axis=0, keepdims=True)
                faiss.normalize_L2(query_vec)
                
                # Find nearest Nan Yar passage (only from available)
                similarities = query_vec @ nan_yar_embeddings.T
                # Filter to available indices and get top-3
                available_sims = [(similarities[0][i], i) for i in available_indices]
                available_sims.sort(reverse=True)
                top_k = min(3, len(available_sims))
                top_indices = [idx for _, idx in available_sims[:top_k]]
                chosen_idx = random.choice(top_indices)
                
                logger.info(f"Personalized Ramana passage selection for {session_id[:8]}... "
                           f"(sim={similarities[0][chosen_idx]:.3f}, idx={chosen_idx})")
                mark_nan_yar_shown(session_id, chosen_idx)
                return nan_yar_passages[chosen_idx]
            except Exception as e:
                logger.warning(f"Personalized Ramana passage selection failed: {e}")
    
    # Random selection from available
    chosen_idx = random.choice(available_indices)
    mark_nan_yar_shown(session_id, chosen_idx)
    logger.info(f"Random Ramana passage selection for {session_id[:8]}... (idx={chosen_idx})")
    return nan_yar_passages[chosen_idx]


def initialize_generator():
    """Initialize ContemplativeGenerator with RAG provider and critic."""
    global generator
    
    try:
        # Determine critic model
        critic_model = llm_model
        if llm_backend == "openrouter" and not critic_model:
            critic_model = "anthropic/claude-sonnet-4"
        
        critic = LocalCritic(
            backend=llm_backend,
            base_url=llm_url if llm_backend == "local" else None,
            model=critic_model,
            threshold=6.0,
        )
        
        # Determine RAG model
        rag_model = llm_model
        if llm_backend == "openrouter" and not rag_model:
            rag_model = "anthropic/claude-sonnet-4"
        
        rag_provider = ContemplativeRAGProvider(
            qa_jsonl_path="./ramana/src/ramana_qa_training.jsonl",
            backend=llm_backend,
            local_url=llm_url if llm_backend == "local" else None,
            model=rag_model,
        )
        
        generator = ContemplativeGenerator(
            inference_provider=rag_provider,
            critic=critic,
        )
        logger.info(f"ContemplativeGenerator initialized successfully (backend={llm_backend})")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        generator = None


def initialize_filtered_passages_rag():
    """
    Initialize filtered passages RAG system.
    
    Reads from filtered_guten/filtered_passages/corpus.jsonl, which is the
    output from filter_passages.py (Stage 4 of the pipeline).
    """
    global filtered_passages_rag
    
    corpus_path = Path(__file__).parent / "filtered_guten" / "filtered_passages" / "corpus.jsonl"
    
    if not corpus_path.exists():
        logger.warning(f"Filtered passages corpus not found: {corpus_path}")
        logger.warning("Run filter_passages.py to generate the corpus. RAG will return empty results.")
        filtered_passages_rag = None
        return
    
    try:
        # NOTE: Unlike ContemplativeRAGProvider (which gets local_url=None for
        # OpenRouter), FilteredPassagesRAG always receives llm_url directly.
        # This is intentional — it uses llm_url for both backends.
        filtered_passages_rag = FilteredPassagesRAG(
            corpus_path=str(corpus_path),
            llm_url=llm_url,
            llm_model=llm_model,
            llm_backend=llm_backend,
        )
        logger.info(f"FilteredPassagesRAG initialized successfully from {corpus_path} (backend={llm_backend})")
    except Exception as e:
        logger.error(f"Failed to initialize FilteredPassagesRAG: {e}")
        filtered_passages_rag = None


# ── Conversation History Builder ───────────────────────────────────────

def _build_conversation_history(session_id: str, mode: str = "direct") -> list[dict]:
    """
    Build conversation_history for the generator.
    
    Includes:
    - In conversational mode: warm system prompt
    - User model threads as a light-touch system hint (if available)
    - Last 2-3 turns as user/assistant pairs
    """
    history = []
    
    # In conversational mode, prepend the warm system prompt
    if mode == "conversational":
        history.append({
            "role": "system",
            "content": CONVERSATIONAL_SYSTEM,
        })
    
    # User model as light-touch context
    threads = get_user_model_threads(session_id)
    if threads:
        threads_text = "; ".join(threads)
        history.append({
            "role": "system",
            "content": (
                f"The visitor has been exploring these contemplative territories: "
                f"{threads_text}. "
                f"You may adjust the depth and vocabulary of your response "
                f"accordingly, but respond to their question as asked. "
                f"Do not assume context they haven't provided."
            ),
        })
    
    # Recent turns as conversation context
    session_history = load_session_history(session_id)
    recent = session_history[-3:]
    for turn in recent:
        q = turn.get("user_input", "")
        r = turn.get("response", "")
        if q:
            history.append({"role": "user", "content": q})
        if r:
            history.append({"role": "assistant", "content": r})
    
    return history if history else None


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    session_id = get_or_create_session_id()
    resp = make_response(render_template('index.html'))
    return set_session_cookie(resp, session_id)


@app.route('/api/nan-yar', methods=['GET'])
def get_nan_yar():
    """Get a Ramana passage (from Nan Yar, Ulladu Narpadu, or Upadesa Undiyar), personalized if user model exists."""
    if not nan_yar_passages:
        return jsonify({'error': 'No Ramana passages loaded'}), 500
    
    session_id = get_or_create_session_id()
    
    # Check for stale session and trigger rollup if needed
    check_and_rollup_stale_session(session_id)
    
    passage_data = select_nan_yar_passage(session_id)
    resp = make_response(jsonify({
        'passage': passage_data.get('text', ''),
        'source': passage_data.get('source', '')
    }))
    return set_session_cookie(resp, session_id)


@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Generate initial response only. Returns immediately.
    Expand and passages are fetched separately by the frontend.
    """
    if generator is None:
        return jsonify({'error': 'Generator not initialized'}), 500
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query field'}), 400
    
    user_query = data['query'].strip()
    if not user_query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    mode = data.get('mode', 'direct')
    if mode not in ('direct', 'conversational'):
        mode = 'direct'
    
    session_id = get_or_create_session_id()
    
    # Check for stale session and trigger rollup if needed
    check_and_rollup_stale_session(session_id)
    
    # Build conversation history for the generator
    conversation_history = _build_conversation_history(session_id, mode=mode)
    
    try:
        result = generator.generate(user_query, conversation_history=conversation_history, mode=mode)
        response_text = result.get('response', '')
        is_silence = result.get('is_silence', False)
        
        if is_silence or not response_text or response_text == '[silence]':
            # In conversational mode, replace silence with a gentle acknowledgment
            if mode == "conversational":
                response_text = (
                    "That question touches something deep. Rather than fill this "
                    "space with words, I'd invite you to sit with it. What do you "
                    "notice when you simply stay with the question?"
                )
                is_silence = False
            else:
                # Record silence turn
                append_turn(session_id, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_input": user_query,
                    "response": "",
                    "is_silence": True,
                })
                
                resp = jsonify({
                    'response': '',
                    'is_silence': True,
                    'message': 'Silence is sometimes the most appropriate response.',
                })
                return set_session_cookie(resp, session_id)
        
        # Record turn (expand + passages will update later)
        append_turn(session_id, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_input": user_query,
            "response": response_text,
            "is_silence": False,
        })
        
        resp = jsonify({
            'response': response_text,
            'is_silence': False,
        })
        return set_session_cookie(resp, session_id)
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/expand', methods=['POST'])
def handle_expand():
    """
    Generate expanded response with post-filter.
    
    If the expanded response fails the coherence filter, retries once
    with the filter's rewrite instruction. Returns blank if both fail.
    """
    if generator is None:
        return jsonify({'error': 'Generator not initialized'}), 500
    
    data = request.get_json()
    if not data or 'query' not in data or 'response' not in data:
        return jsonify({'error': 'Missing query or response field'}), 400
    
    user_query = data['query'].strip()
    response_text = data['response'].strip()
    mode = data.get('mode', 'direct')
    if mode not in ('direct', 'conversational'):
        mode = 'direct'
    
    if not user_query or not response_text:
        return jsonify({'expanded_response': ''})
    
    session_id = get_or_create_session_id()
    user_context = get_user_context_for_prompt(session_id)
    
    MIN_CONFIDENCE = 0.5
    
    try:
        # First attempt
        expanded_response = generator.expand(user_query, response_text, mode=mode)
        if not expanded_response:
            return jsonify({'expanded_response': ''})
        
        # Post-filter
        verdict = filter_expand_with_llm(
            user_input=user_query,
            response=response_text,
            expanded_response=expanded_response,
            user_context=user_context,
        )
        logger.info(f"Expand filter: acceptable={verdict['acceptable']}, "
                    f"confidence={verdict['confidence']:.2f}, "
                    f"reason={verdict['reason']}")
        
        if verdict["acceptable"] and verdict["confidence"] >= MIN_CONFIDENCE:
            return jsonify({'expanded_response': expanded_response})
        
        # Failed — retry once with rewrite instruction
        rewrite_hint = verdict.get("rewrite_instruction") or verdict["reason"]
        logger.info(f"Expand rejected, retrying with hint: {rewrite_hint}")
        
        # Augment the query with the rewrite instruction
        augmented_query = (
            f"{user_query}\n\n"
            f"[Note: A previous expansion was rejected because: {rewrite_hint}. "
            f"Please generate a new expansion that avoids this issue.]"
        )
        
        expanded_response_2 = generator.expand(augmented_query, response_text, mode=mode)
        if not expanded_response_2:
            logger.info("Retry expand returned empty, returning blank")
            return jsonify({'expanded_response': ''})
        
        # Post-filter the retry
        verdict_2 = filter_expand_with_llm(
            user_input=user_query,
            response=response_text,
            expanded_response=expanded_response_2,
            user_context=user_context,
        )
        logger.info(f"Expand retry filter: acceptable={verdict_2['acceptable']}, "
                    f"confidence={verdict_2['confidence']:.2f}, "
                    f"reason={verdict_2['reason']}")
        
        if verdict_2["acceptable"] and verdict_2["confidence"] >= MIN_CONFIDENCE:
            return jsonify({'expanded_response': expanded_response_2})
        
        # Both attempts failed — return blank
        logger.info("Expand failed post-filter twice, returning blank")
        return jsonify({'expanded_response': ''})
    
    except Exception as e:
        logger.error(f"Error expanding response: {e}", exc_info=True)
        return jsonify({'error': f'Expand failed: {str(e)}'}), 500


@app.route('/api/retrieve-passages', methods=['POST'])
def retrieve_passages():
    """
    Retrieve and filter passages. Called after expand returns.
    
    Over-retrieves (top_k=8), then LLM-filters each passage for
    coherence/alignment, returns the single best passage (or none).
    """
    logger.info("Received retrieve-passages request")
    
    if filtered_passages_rag is None:
        logger.warning("FilteredPassagesRAG not initialized")
        return jsonify({'passages': []})
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400
    
    user_input = data.get('user_input', '')
    response = data.get('response', '')
    expanded_response = data.get('expanded_response', '')
    
    if not user_input and not response:
        return jsonify({'passages': []})
    
    session_id = get_or_create_session_id()
    user_context = get_user_context_for_prompt(session_id)
    
    try:
        # Over-retrieve candidates
        candidates = filtered_passages_rag.retrieve_with_rewrite(
            user_input=user_input,
            response=response,
            expanded_response=expanded_response,
            user_context=user_context,
            top_k=8,
        )
        logger.info(f"Retrieved {len(candidates)} candidate passages")
        
        if not candidates:
            return jsonify({'passages': []})
        
        # Filter out already-shown passages
        shown_passage_ids = get_shown_sidebar_passage_ids(session_id)
        available_candidates = [
            c for c in candidates 
            if c.get("passage_id") not in shown_passage_ids
        ]
        
        if not available_candidates:
            logger.info(f"All candidate passages already shown for {session_id[:8]}...")
            # If all candidates were shown, allow repeats (but still prefer unshown if any)
            available_candidates = candidates
        
        # LLM post-filter: rate each candidate
        best_passage = None
        best_confidence = 0.0
        
        for candidate in available_candidates:
            verdict = filter_passage_with_llm(
                passage=candidate,
                user_input=user_input,
                response=response,
                expanded_response=expanded_response,
                user_context=user_context,
            )
            
            pid = candidate.get("passage_id", "?")
            logger.info(f"  Filter {pid}: show={verdict['show']}, "
                       f"confidence={verdict['confidence']:.2f}, "
                       f"reason={verdict['reason']}")
            
            if verdict["show"] and verdict["confidence"] > best_confidence:
                best_confidence = verdict["confidence"]
                best_passage = candidate
                best_passage["_filter_reason"] = verdict["reason"]
                best_passage["_filter_confidence"] = verdict["confidence"]
        
        # Apply minimum confidence threshold
        MIN_CONFIDENCE = 0.5
        if best_passage and best_confidence >= MIN_CONFIDENCE:
            passage_id = best_passage.get('passage_id')
            mark_sidebar_passage_shown(session_id, passage_id)
            
            formatted = {
                'passage_id': passage_id,
                'text': best_passage.get('text', ''),
                'author': best_passage.get('author', 'Unknown'),
                'title': best_passage.get('title', 'Untitled'),
                'tradition': best_passage.get('tradition', 'unknown'),
                'themes': best_passage.get('themes', []),
                'similarity': best_passage.get('similarity', 0.0),
            }
            logger.info(f"Selected passage {formatted['passage_id']} "
                       f"(confidence={best_confidence:.2f})")
            return jsonify({'passages': [formatted]})
        else:
            logger.info(f"No passage met confidence threshold "
                       f"(best={best_confidence:.2f}, min={MIN_CONFIDENCE})")
            return jsonify({'passages': []})
    
    except Exception as e:
        logger.error(f"Error retrieving passages: {e}", exc_info=True)
        return jsonify({'error': f'Retrieval failed: {str(e)}', 'passages': []}), 500


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ramana Maharshi Website API Server")
    parser.add_argument(
        '--llm-backend',
        type=str,
        default=os.environ.get('LLM_BACKEND', 'local'),
        choices=['local', 'openrouter'],
        help='LLM backend: "local" (vLLM) or "openrouter" (default: local, or LLM_BACKEND env var)'
    )
    parser.add_argument(
        '--llm-url',
        type=str,
        default=os.environ.get('LLM_URL', 'http://localhost:5000/v1'),
        help='LLM API URL (default: http://localhost:5000/v1 for local, or LLM_URL env var)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=os.environ.get('LLM_MODEL', None),
        help='LLM model name (required for OpenRouter, auto-detected for local, or LLM_MODEL env var)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('PORT', '5001')),
        help='Port to run Flask app on (default: 5001, or PORT env var)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=os.environ.get('HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0, or HOST env var)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        help='Enable Flask debug mode (or FLASK_DEBUG=true env var)'
    )
    
    args = parser.parse_args()
    
    # Set module-level LLM configuration (no 'global' needed at module scope)
    llm_backend = args.llm_backend
    llm_url = args.llm_url
    llm_model = args.llm_model
    
    # Auto-set OpenRouter URL if backend is openrouter and URL is still default
    if llm_backend == 'openrouter' and llm_url == 'http://localhost:5000/v1':
        llm_url = 'https://openrouter.ai/api/v1'
        logger.info(f"Auto-set LLM URL to OpenRouter: {llm_url}")
    
    # Validate OpenRouter configuration
    if llm_backend == 'openrouter':
        if not os.environ.get('OPENROUTER_API_KEY'):
            logger.warning("OPENROUTER_API_KEY not set - OpenRouter calls will fail")
        if not llm_model:
            logger.info("No model specified for OpenRouter, will use default: anthropic/claude-sonnet-4")
    
    logger.info(f"LLM Configuration: backend={llm_backend}, url={llm_url}, model={llm_model or 'auto'}")
    
    logger.info("Initializing Ramana API server...")
    load_nan_yar_passages()
    # Detect model early so llm_model_name is set before filter functions need it
    detect_llm_model()
    if llm_model_name == "unknown":
        logger.error("LLM model detection failed and no model specified. Filter functions will fail.")
        logger.error("Please specify --llm-model or ensure local vLLM is running.")
    initialize_generator()
    initialize_filtered_passages_rag()
    
    # Ensure sessions directory exists
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Disable reloader under debugpy to avoid SystemExit(3) conflict
    import sys
    under_debugger = "debugpy" in sys.modules
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=not under_debugger and args.debug)
