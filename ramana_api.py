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

import json
import logging
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

# ── Constants ─────────────────────────────────────────────────────────

SESSIONS_DIR = Path(__file__).parent / "sessions"
LLM_URL = "http://localhost:5000/v1"

# Session cookie settings
SESSION_COOKIE_NAME = "ramana_session"
SESSION_COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 year

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

Respond with ONLY a JSON object:
{{
  "show": true or false,
  "confidence": <float 0.0 to 1.0>,
  "reason": "<one sentence>"
}}

No markdown, no preamble."""

# Session rollup prompt: generates a user model from conversation history
SESSION_ROLLUP_PROMPT = """You are maintaining a contemplative interest profile for a visitor to a website devoted to Ramana Maharshi's teachings. Given this visitor's conversation history (and optionally their previous profile), produce an updated set of semantic threads that capture the contemplative territory this person is drawn to.

Each thread should be a brief description (under 30 words) of an experiential or contemplative interest, written in clear, tradition-neutral language. Focus on what territory of awareness, self, mind, realization, or lived contemplative experience they are exploring — NOT surface topics or conversation mechanics.

GOOD thread: "Inquiry into whether effort itself is an obstacle — the paradox of trying to stop trying."
BAD thread: "The user frequently asks about meditation techniques."

Produce 3-7 threads that form a coherent portrait of this visitor's contemplative engagement. Threads should evolve: if earlier interests have faded in recent conversations, let them go. If new territory has emerged, include it.

{previous_profile_section}
CONVERSATION HISTORY:
{conversation_history}

Respond with ONLY a JSON object:
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
        {"semantic_threads": [...], "themes": [...]} or empty dict
    """
    rollup_file = get_rollup_file(session_id)
    if not rollup_file.exists():
        return {}
    try:
        with open(rollup_file, "r") as f:
            return json.loads(f.read())
    except (json.JSONDecodeError, IOError):
        return {}


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
        resp = llm_http.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            raw = raw.strip()
        
        result = json.loads(raw)
        
        rollup = {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": len(history),
            "semantic_threads": result.get("semantic_threads", []),
            "themes": result.get("themes", []),
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
        # Write a minimal rollup so we don't retry constantly
        rollup = {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": len(history),
            "semantic_threads": prev_threads,  # Keep previous if LLM fails
            "themes": existing_model.get("themes", []),
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
        resp = llm_http.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.1,
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            raw = raw.strip()
        
        result = json.loads(raw)
        
        return {
            "show": bool(result.get("show", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    
    except Exception as e:
        logger.error(f"Passage filter LLM call failed: {e}")
        # On failure, allow the passage through with low confidence
        return {"show": True, "confidence": 0.3, "reason": "filter unavailable"}


def detect_llm_model():
    """Auto-detect model from local vLLM API."""
    global llm_model_name
    try:
        resp = llm_http.get(f"{LLM_URL}/models")
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            llm_model_name = data["data"][0]["id"]
            logger.info(f"Auto-detected LLM model: {llm_model_name}")
        else:
            llm_model_name = "unknown"
    except Exception as e:
        logger.warning(f"Could not detect LLM model: {e}")
        llm_model_name = "unknown"


# ── Initialization ────────────────────────────────────────────────────

def load_nan_yar_passages():
    """Load Nan_Yar passages from file and pre-embed for personalized selection."""
    global nan_yar_passages, nan_yar_embeddings, nan_yar_embedder
    
    nan_yar_path = Path(__file__).parent / "ramana" / "nan-yar.txt"
    if not nan_yar_path.exists():
        logger.warning(f"Nan_Yar file not found: {nan_yar_path}")
        nan_yar_passages = []
        return
    
    with open(nan_yar_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    passages = [p.strip() for p in content.split('\n\n') if p.strip()]
    if not passages:
        passages = [p.strip() for p in content.split('\n') if p.strip()]
    
    nan_yar_passages = passages
    logger.info(f"Loaded {len(nan_yar_passages)} Nan_Yar passages")
    
    # Pre-embed for personalized selection
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
        
        nan_yar_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = nan_yar_embedder.encode(nan_yar_passages, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        nan_yar_embeddings = embeddings
        logger.info(f"Pre-embedded {len(nan_yar_passages)} Nan_Yar passages for personalized selection")
    except Exception as e:
        logger.warning(f"Could not pre-embed Nan_Yar passages (will use random): {e}")
        nan_yar_embeddings = None
        nan_yar_embedder = None


def select_nan_yar_passage(session_id: str) -> str:
    """
    Select a Nan Yar passage. Personalized if user model exists,
    otherwise random.
    """
    if not nan_yar_passages:
        return ""
    
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
                
                # Find nearest Nan Yar passage
                similarities = query_vec @ nan_yar_embeddings.T
                # Pick from top-3 randomly for some variety
                top_k = min(3, len(nan_yar_passages))
                top_indices = np.argsort(similarities[0])[-top_k:][::-1]
                chosen_idx = random.choice(top_indices.tolist())
                
                logger.info(f"Personalized Nan_Yar selection for {session_id[:8]}... "
                           f"(sim={similarities[0][chosen_idx]:.3f})")
                return nan_yar_passages[chosen_idx]
            except Exception as e:
                logger.warning(f"Personalized Nan_Yar selection failed: {e}")
    
    return random.choice(nan_yar_passages)


def initialize_generator():
    """Initialize ContemplativeGenerator with RAG provider and critic."""
    global generator
    
    try:
        critic = LocalCritic(
            backend="local",
            base_url="http://localhost:5000/v1",
            threshold=6.0,
        )
        rag_provider = ContemplativeRAGProvider(
            commentaries_path="./ramana/Commentaries_qa_excert.txt",
            backend="local",
            local_url="http://localhost:5000/v1",
        )
        generator = ContemplativeGenerator(
            inference_provider=rag_provider,
            critic=critic,
        )
        logger.info("ContemplativeGenerator initialized successfully")
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
        filtered_passages_rag = FilteredPassagesRAG(
            corpus_path=str(corpus_path),
            llm_url="http://localhost:5000/v1",
        )
        logger.info(f"FilteredPassagesRAG initialized successfully from {corpus_path}")
    except Exception as e:
        logger.error(f"Failed to initialize FilteredPassagesRAG: {e}")
        filtered_passages_rag = None


# ── Conversation History Builder ───────────────────────────────────────

def _build_conversation_history(session_id: str) -> list[dict]:
    """
    Build conversation_history for the generator.
    
    Includes:
    - User model threads as a light-touch system hint (if available)
    - Last 2-3 turns as user/assistant pairs
    """
    history = []
    
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
    """Get a Nan_Yar passage, personalized if user model exists."""
    if not nan_yar_passages:
        return jsonify({'error': 'No Nan_Yar passages loaded'}), 500
    
    session_id = get_or_create_session_id()
    
    # Check for stale session and trigger rollup if needed
    check_and_rollup_stale_session(session_id)
    
    passage = select_nan_yar_passage(session_id)
    return jsonify({'passage': passage})


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
    
    session_id = get_or_create_session_id()
    
    # Check for stale session and trigger rollup if needed
    check_and_rollup_stale_session(session_id)
    
    # Build conversation history for the generator
    conversation_history = _build_conversation_history(session_id)
    
    try:
        result = generator.generate(user_query, conversation_history=conversation_history)
        response_text = result.get('response', '')
        is_silence = result.get('is_silence', False)
        
        if is_silence or not response_text or response_text == '[silence]':
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
    Generate expanded response. Called after /api/query returns.
    """
    if generator is None:
        return jsonify({'error': 'Generator not initialized'}), 500
    
    data = request.get_json()
    if not data or 'query' not in data or 'response' not in data:
        return jsonify({'error': 'Missing query or response field'}), 400
    
    user_query = data['query'].strip()
    response_text = data['response'].strip()
    
    if not user_query or not response_text:
        return jsonify({'expanded_response': ''})
    
    try:
        expanded_response = generator.expand(user_query, response_text)
        if not expanded_response:
            expanded_response = ''
        
        return jsonify({'expanded_response': expanded_response})
    
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
        
        # LLM post-filter: rate each candidate
        best_passage = None
        best_confidence = 0.0
        
        for candidate in candidates:
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
            formatted = {
                'passage_id': best_passage.get('passage_id'),
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
    logger.info("Initializing Ramana API server...")
    load_nan_yar_passages()
    initialize_generator()
    initialize_filtered_passages_rag()
    detect_llm_model()
    
    # Ensure sessions directory exists
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Disable reloader under debugpy to avoid SystemExit(3) conflict
    import sys
    under_debugger = "debugpy" in sys.modules
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=not under_debugger)
