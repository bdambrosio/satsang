"""
Two-pass contemplative relevance filter for extracted passages.

Pass 1: Embedding distance — cheap, deterministic.
  Compares each passage's semantic_threads against theme glosses.
  Passages below REJECT_THRESHOLD are filtered out immediately.
  Passages above ACCEPT_THRESHOLD are accepted immediately.

Pass 2: LLM judgment — full passage text sent for borderline cases.
  Everything between the two thresholds gets a yes/no call with the
  actual passage text, not just the thread descriptions.

Incremental: reads any existing filtered_passages/corpus.jsonl to skip already-filtered
passages.  Each run writes a timestamped file under filtered_passages/runs/,
then rebuilds the merged filtered_passages/corpus.jsonl from all run files.

Input: reads from filtered_guten/passages/corpus.jsonl by default (output from passage_identification.py).
Output: writes to filtered_guten/filtered_passages/runs/ and rebuilds filtered_guten/filtered_passages/corpus.jsonl.

Usage:
  python filter_passages.py
  python filter_passages.py --reject 0.35 --accept 0.55
  python filter_passages.py --no-llm
  python filter_passages.py --passages custom_input.jsonl  # override default input
"""

import json
import argparse
import numpy as np
import requests
from datetime import datetime, timezone
from pathlib import Path
from sentence_transformers import SentenceTransformer

VLLM_BASE = "http://0.0.0.0:5000/v1"

# ── Thresholds ─────────────────────────────────────────────────────────
DEFAULT_REJECT = 0.35
DEFAULT_ACCEPT = 0.55

# ── Input/Output paths ───────────────────────────────────────────────────────
PASSAGES_INPUT = Path("filtered_guten/passages/corpus.jsonl")
OUTPUT_DIR = Path("filtered_guten/filtered_passages")
RUNS_DIR = OUTPUT_DIR / "runs"
CORPUS_FILE = OUTPUT_DIR / "corpus.jsonl"

# ── Theme glosses ──────────────────────────────────────────────────────

THEME_GLOSSES = {
    "self-inquiry": "Investigating the nature of the self, asking 'Who am I?', turning attention inward to find the source of awareness.",
    "ego-dissolution": "The falling away of the separate self, dissolution of the ego and personal identity into undivided being.",
    "true-nature": "Recognition of one's original nature, essential being, the Self beyond personality and conditioning.",
    "witness-awareness": "Pure awareness observing experience without identification, the silent witness behind all perception.",
    "surrender": "Letting go of personal will and control, yielding to what is, abandoning the illusion of the doer.",
    "effortless-effort": "Action arising spontaneously without the sense of a doer, wu wei, grace-filled activity beyond striving.",
    "beyond-knowledge": "The limits of conceptual knowledge, that which cannot be grasped by the mind, unknowing as gateway to truth.",
    "direct-experience": "Immediate unmediated knowing, experiential realization rather than intellectual understanding.",
    "nature-of-mind": "The essential character of consciousness, mind examining itself, awareness recognizing its own nature.",
    "stillness": "Inner silence, cessation of mental activity, the peace and quietude underlying all experience.",
    "attention": "The quality and direction of awareness, mindfulness, the practice of sustained noticing.",
    "detachment": "Non-clinging, freedom from attachment to outcomes, objects, and mental states, inner renunciation.",
    "illusion": "Maya, the unreality of appearances, mistaking the phenomenal world for ultimate reality.",
    "unity": "Non-duality, the oneness underlying apparent multiplicity, the collapse of subject-object division.",
    "ordinary-life": "Awakening expressed in everyday activity, the sacred in the mundane, chopping wood carrying water.",
    "impermanence": "The transient nature of all phenomena, change as fundamental reality, nothing endures.",
    "suffering": "The nature of suffering, its causes, and its relationship to identification and craving.",
    "desire": "The role of wanting and craving in binding consciousness, freedom from desire as liberation.",
    "liberation": "Moksha, nirvana, freedom from bondage, the end of seeking, final realization.",
    "death": "Physical death, ego death, the deathless, dying before death as spiritual practice.",
    "guru-grace": "The role of the teacher, transmission beyond words, grace as catalyst for awakening.",
    "devotion": "Bhakti, love as a path to the divine, the heart's surrender to the beloved.",
    "compassion": "Universal compassion, karuna, the natural expression of realized being toward all creatures.",
    "the-source": "The origin of all manifestation, the ground of being, that from which everything arises and returns.",
    "presence": "Being fully here now, immediate presence prior to thought, the felt sense of existence.",
    "the-sacred": "The numinous, the holy, encounter with the transcendent or the mysterium tremendum.",
    "practice-as-life": "Spiritual practice integrated into all activity, life itself as the path, no separation between practice and living.",
    "paradox": "Spiritual paradox, holding contradictions, the coincidence of opposites in realization.",
    "body": "The body as vehicle for awareness, embodiment, somatic experience in contemplative practice.",
    "scripture-and-tradition": "The role of sacred texts and lineage, relationship between teaching and direct experience.",
}

# ── LLM filter prompt ──────────────────────────────────────────────────

LLM_SYSTEM = """You are an expert contemplative literature editor curating passages for a website devoted to the teachings of Ramana Maharshi. You will be given a passage extracted from a text, along with its metadata. Your task is to judge whether this passage is genuinely contemplatively relevant — whether it speaks to inner experience, spiritual realization, the nature of mind or self, or the territory pointed to by contemplative traditions.

A passage may USE contemplative-sounding language (impermanence, paradox, unity) while actually discussing logic, physics, aesthetics, biography, or landscape description. That is NOT sufficient. The passage must speak FROM or ABOUT contemplative territory — the lived experience of awareness, realization, self-knowledge, liberation, or the sacred.

Borderline cases to consider:
- Philosophy OF mind (analytical) vs philosophy FROM mind (contemplative)
- Landscape description (scenic) vs nature as mirror of inner stillness (contemplative)
- Ethical instruction (moral) vs ethical insight arising from realization (contemplative)
- Aesthetic observation (artistic) vs art as encounter with the numinous (contemplative)

Respond with ONLY a JSON object:
{
  "relevant": true or false,
  "confidence": "high" or "medium",
  "reason": "<one sentence explaining your judgment>"
}

No markdown, no preamble."""

LLM_USER_TEMPLATE = """Passage from: {title} by {author}
Tradition: {tradition}
Orientation: {orientation}
Tagged themes: {themes}

--- PASSAGE TEXT ---
{text}
--- END ---

Is this passage genuinely contemplatively relevant?"""


def get_model_name():
    try:
        resp = requests.get(f"{VLLM_BASE}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
        name = models[0]["id"]
        print(f"LLM model: {name}")
        return name
    except Exception as e:
        print(f"Could not query vLLM models endpoint: {e}")
        print("LLM pass will be skipped for borderline passages.")
        return None


def llm_judge(passage: dict, model_name: str) -> dict | None:
    """Ask the LLM whether a passage is genuinely contemplative."""
    user_msg = LLM_USER_TEMPLATE.format(
        title=passage.get("title", "?"),
        author=passage.get("author", "?"),
        tradition=passage.get("tradition", "?"),
        orientation=passage.get("orientation", "?"),
        themes=passage.get("themes", []),
        text=passage.get("text", ""),
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(
            f"{VLLM_BASE}/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        response_data = resp.json()
        
        # Check response structure
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            print(f"    LLM error: Response has no choices")
            return None
        
        message = response_data["choices"][0].get("message", {})
        content = message.get("content")
        
        if content is None:
            print(f"    LLM error: Response has no content field")
            return None
        
        raw = content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        return json.loads(raw)

    except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
        print(f"    LLM error: {e}")
        return None


def load_passages(path: str) -> list[dict]:
    passages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


def load_existing_ids() -> set[str]:
    """Load passage_ids already present in corpus.jsonl (if it exists)."""
    ids = set()
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        pid = rec.get("passage_id")
                        if pid:
                            ids.add(pid)
                    except json.JSONDecodeError:
                        continue
    return ids


def rebuild_corpus():
    """Rebuild corpus.jsonl by merging all run files, deduplicating by passage_id."""
    seen = set()
    records = []
    if not RUNS_DIR.exists():
        return
    for run_file in sorted(RUNS_DIR.glob("*.jsonl")):
        with open(run_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("passage_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    records.append(line)
    with open(CORPUS_FILE, "w") as f:
        for line in records:
            f.write(line + "\n")
    return len(records)


def clean_record(p: dict) -> dict:
    """Return passage record without internal scoring fields."""
    return {k: v for k, v in p.items() if not k.startswith("_")}


def compute_best_similarity(threads: list[str], model, theme_vecs, theme_names):
    """Return (best_sim, best_theme, per-thread details)."""
    if not threads:
        return 0.0, "none", []

    thread_vecs = model.encode(threads, normalize_embeddings=True)
    sims = thread_vecs @ theme_vecs.T

    details = []
    passage_best = 0.0
    passage_best_theme = ""

    for i, thread in enumerate(threads):
        top3_idx = np.argsort(sims[i])[-3:][::-1]
        top3 = [(theme_names[j], float(sims[i][j])) for j in top3_idx]
        best_sim = top3[0][1]
        if best_sim > passage_best:
            passage_best = best_sim
            passage_best_theme = top3[0][0]
        details.append((thread, top3))

    return passage_best, passage_best_theme, details


def main():
    parser = argparse.ArgumentParser(description="Two-pass contemplative relevance filter")
    parser.add_argument("--passages", default=str(PASSAGES_INPUT),
                        help=f"Path to passages JSONL (default: {PASSAGES_INPUT})")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--reject", type=float, default=DEFAULT_REJECT,
                        help=f"Embedding threshold below which passages are rejected (default {DEFAULT_REJECT})")
    parser.add_argument("--accept", type=float, default=DEFAULT_ACCEPT,
                        help=f"Embedding threshold above which passages are accepted (default {DEFAULT_ACCEPT})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM pass, just show embedding bins")
    args = parser.parse_args()

    passages_path = Path(args.passages)
    if not passages_path.exists():
        print(f"Input file not found: {passages_path}")
        print(f"Run passage_identification.py first to generate {PASSAGES_INPUT}")
        return

    # ── Load existing corpus for incremental skip ─────────────────────
    existing_ids = load_existing_ids()
    if existing_ids:
        print(f"Found {len(existing_ids)} passages already in {CORPUS_FILE}")

    all_passages = load_passages(str(passages_path))
    print(f"Loaded {len(all_passages)} passages from {passages_path}")
    
    passages = [p for p in all_passages if p.get("passage_id") not in existing_ids]
    skipped = len(all_passages) - len(passages)

    if skipped:
        print(f"Skipping {skipped} already-filtered passages")
    if not passages:
        print("All passages already filtered. Nothing to do.")
        return

    print(f"{len(passages)} passages to filter")

    # ── Setup ─────────────────────────────────────────────────────────
    print(f"Loading embedding model: {args.model}")
    emb_model = SentenceTransformer(args.model)

    theme_names = list(THEME_GLOSSES.keys())
    theme_texts = list(THEME_GLOSSES.values())
    print(f"Embedding {len(theme_names)} theme glosses...")
    theme_vecs = emb_model.encode(theme_texts, normalize_embeddings=True)

    print(f"Thresholds: reject < {args.reject:.2f} | borderline | accept > {args.accept:.2f}\n")

    # ── Pass 1: Embedding ──────────────────────────────────────────────
    rejected = []
    accepted = []
    borderline = []

    for p in passages:
        threads = p.get("semantic_threads", [])
        best_sim, best_theme, details = compute_best_similarity(
            threads, emb_model, theme_vecs, theme_names
        )
        p["_best_sim"] = best_sim
        p["_best_theme"] = best_theme
        p["_thread_details"] = details

        if best_sim < args.reject:
            rejected.append(p)
        elif best_sim > args.accept:
            accepted.append(p)
        else:
            borderline.append(p)

    print("=" * 70)
    print("PASS 1 — EMBEDDING FILTER")
    print(f"  Rejected  (sim < {args.reject:.2f}): {len(rejected)}")
    print(f"  Borderline ({args.reject:.2f}–{args.accept:.2f}): {len(borderline)}")
    print(f"  Accepted  (sim > {args.accept:.2f}): {len(accepted)}")
    print()

    # ── Pass 2: LLM on borderline ─────────────────────────────────────
    llm_accepted = []
    llm_rejected = []
    llm_failed = []
    
    if not borderline:
        print("No borderline passages — nothing for LLM to judge.")
    elif args.no_llm:
        print("(--no-llm flag set, skipping LLM pass)")
    else:
        model_name = get_model_name()
        if model_name is not None:
            print("=" * 70)
            print(f"PASS 2 — LLM JUDGMENT on {len(borderline)} borderline passages\n")

            for p in sorted(borderline, key=lambda x: x["_best_sim"]):
                pid = p.get("passage_id", "?")
                result = llm_judge(p, model_name)

                if result is None:
                    llm_failed.append(p)
                    print(f"  {pid:16s} sim={p['_best_sim']:.3f}  LLM: FAILED")
                    continue

                relevant = result.get("relevant", False)
                reason = result.get("reason", "")
                conf = result.get("confidence", "?")

                if relevant:
                    llm_accepted.append(p)
                    tag = "✓ KEEP"
                else:
                    llm_rejected.append(p)
                    tag = "✗ DROP"

                print(f"  {pid:16s} sim={p['_best_sim']:.3f}  LLM: {tag} ({conf})")
                print(f"                   {reason}")

    # ── Write accepted passages to run file ───────────────────────────
    all_kept = accepted + llm_accepted
    all_dropped = rejected + llm_rejected

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_file = RUNS_DIR / f"{timestamp}.jsonl"

    with open(run_file, "w") as f:
        for p in all_kept:
            f.write(json.dumps(clean_record(p)) + "\n")

    print()
    print("=" * 70)
    print("RESULTS")
    print(f"  Accepted (embedding):    {len(accepted)}")
    print(f"  Accepted (LLM):          {len(llm_accepted)}")
    print(f"  Rejected (embedding):    {len(rejected)}")
    print(f"  Rejected (LLM):          {len(llm_rejected)}")
    if llm_failed:
        print(f"  LLM failures:            {len(llm_failed)}")
    print(f"  ─────────────────────────────")
    print(f"  Total kept:              {len(all_kept)}")
    print(f"  Total filtered:          {len(all_dropped)}")
    print(f"  Run file:                {run_file}")

    # ── Rebuild merged corpus ─────────────────────────────────────────
    total = rebuild_corpus()
    print(f"  Corpus total:            {total} passages in {CORPUS_FILE}")


if __name__ == "__main__":
    main()
