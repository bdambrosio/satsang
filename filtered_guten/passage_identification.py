#!/usr/bin/env python3
"""
Extract self-contained contemplative passages from corpus texts.
Reads annotated JSONL, filters to core/useful + high/medium confidence,
slides through each text and asks LLM to identify standalone passages.
"""

import json
import re
import sys
import time
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path

VLLM_BASE = "http://0.0.0.0:5000/v1"

# Rough chars-per-token estimate for English prose
CHARS_PER_TOKEN = 4

# Window size in tokens for LLM context
WINDOW_TOKENS = 6000
OVERLAP_TOKENS = 1500

SYSTEM_PROMPT = """You are an expert editor curating a cross-tradition contemplative literature corpus for a website devoted to the teachings of Ramana Maharshi. Your task is to identify self-contained passages within a text that could stand alone in a sidebar alongside Ramana's teachings, illuminating the same experiential territory from a different tradition or perspective.

A good passage is:
- SELF-CONTAINED: A reader encountering it with no surrounding context can understand it. It does not begin mid-argument, reference an unclear antecedent, or end on a dependent clause.
- CONTEMPLATIVELY RELEVANT: It speaks to inner experience, spiritual realization, the nature of mind or self, or the territory pointed to by contemplative traditions. Mere ethical advice, biographical detail, historical narrative, or literary description is NOT sufficient.
- FAITHFUL: You extract the EXACT text from the source. Do not paraphrase, summarize, abridge, or alter the wording in any way.
- APPROPRIATELY SIZED: Long enough to convey a complete thought (typically 50-400 words). A single sentence is too short unless it is an aphorism of genuine depth. More than ~400 words suggests the passage could be split.

For DIALOGUE texts: keep question-and-answer pairs together. The question provides essential context for the answer.
For POETRY: extract complete poems or complete stanzas. Never split mid-stanza.
For APHORISMS: individual aphorisms or small thematic clusters.
For PROSE: complete paragraphs or paragraph groups that form a self-contained argument or reflection.
For SCRIPTURE: complete verses or verse groups that form a unit.



You will be given:
- Document metadata (author, title, tradition, orientation, themes)
- A section of the text with line numbers

Respond with a JSON array of passage objects. If no passages in this section meet the criteria, return an empty array [].

Each passage object:
{
  "start_line": <first line number of passage>,
  "end_line": <last line number of passage>,
  "passage_type": "<dialogue|verse|prose|aphorism|prayer|scripture>",
  "themes": ["<theme1>", ...],
  "semantic_threads": ["<brief description of each independent meaning>", ...],
  "standalone_confidence": "<high|medium>",
  "notes": "<optional: context needed, or null>"
}

CRITICAL RULES:
- Return ONLY the JSON array. No markdown, no explanation, no preamble.
- Use EXACT line numbers from the text provided.
- themes must use the controlled vocabulary: self-inquiry, ego-dissolution, true-nature, witness-awareness, surrender, effortless-effort, beyond-knowledge, direct-experience, nature-of-mind, stillness, attention, detachment, illusion, unity, ordinary-life, impermanence, suffering, desire, liberation, death, guru-grace, devotion, compassion, the-source, presence, the-sacred, practice-as-life, paradox, body, scripture-and-tradition
- semantic_threads: 1-3 brief descriptions (under 30 words each) of the contemplative or experiential meaning in this passage. These will be used as embedding keys for retrieval alongside Ramana Maharshi's teachings, so express them in clear, tradition-neutral language that captures the inner or spiritual territory the passage points to. Focus on what the passage says about awareness, self, mind, realization, or lived contemplative experience — NOT on its literary devices, logical structure, rhetorical strategy, or surface topic. Bad example: 'Analogy of darkness versus brightness illustrates true seeing.' Good example: 'Non-reliance on any fixed principle opens perception; clinging to method blinds, while letting go reveals clear seeing.
- Be SELECTIVE. It is far better to return 2 excellent passages than 10 mediocre ones. Most text windows will yield 0-3 passages.
- Passages must not overlap with each other."""


def get_model_name():
    """Query vLLM for the served model name."""
    try:
        resp = requests.get(f"{VLLM_BASE}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
        if len(models) == 1:
            name = models[0]["id"]
            print(f"Found model: {name}")
            return name
        else:
            print(f"Multiple models found: {[m['id'] for m in models]}")
            print(f"Using first: {models[0]['id']}")
            return models[0]["id"]
    except Exception as e:
        print(f"Could not query models endpoint: {e}")
        sys.exit(1)


def load_qualified_entries(annotations_path: Path) -> list[dict]:
    """Load annotations and filter to core/useful + high/medium confidence."""
    entries = []
    with open(annotations_path) as f:
        for line in f:
            rec = json.loads(line)
            if "_error" in rec:
                continue
            rel = rec.get("corpus_relevance", "")
            conf = rec.get("confidence", "")
            if rel in ("core", "useful") and conf in ("high", "medium"):
                entries.append(rec)
    return entries


def load_text_with_lines(text_path: Path) -> list[str]:
    """Load text file and return as list of lines."""
    with open(text_path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def make_windows(lines: list[str], window_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Slide through lines creating overlapping windows.
    Returns list of {start_line, end_line, text} dicts.
    Line numbers are 1-indexed to match what we show the LLM.
    """
    windows = []
    window_chars = window_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    i = 0
    while i < len(lines):
        # Accumulate lines until we hit window size
        char_count = 0
        j = i
        while j < len(lines) and char_count < window_chars:
            char_count += len(lines[j])
            j += 1

        # Build numbered text for this window
        numbered_lines = []
        for k in range(i, j):
            numbered_lines.append(f"{k+1:6d} | {lines[k].rstrip()}")

        windows.append({
            "start_line": i + 1,
            "end_line": j,
            "text": "\n".join(numbered_lines),
        })

        # Advance by window minus overlap
        advance_chars = 0
        next_i = i
        target_advance = window_chars - overlap_chars
        while next_i < j and advance_chars < target_advance:
            advance_chars += len(lines[next_i])
            next_i += 1

        # Ensure we always advance at least one line
        if next_i <= i:
            next_i = i + 1

        i = next_i

    return windows


def build_user_prompt(entry: dict, window: dict,
                      num_windows: int = 1, window_idx: int = 1,
                      max_passages: int = 30) -> str:
    """Construct the user prompt with metadata and text window."""
    meta = (
        f"Document: {entry.get('title', '?')} by {entry.get('author', '?')}\n"
        f"Tradition: {entry.get('tradition', '?')}"
    )
    if entry.get("tradition_secondary"):
        meta += f" (secondary: {entry['tradition_secondary']})"
    meta += (
        f"\nOrientation: {entry.get('orientation', '?')}\n"
        f"Era: {entry.get('era', '?')}\n"
        f"Document themes: {entry.get('themes', [])}\n"
        f"Contemplative depth: {entry.get('contemplative_depth', '?')}\n"
    )

    prompt = (
        f"{meta}\n"
        f"Text section (lines {window['start_line']}-{window['end_line']}):\n\n"
        f"{window['text']}"
    )

    if num_windows > 20:
        prompt += (
            f"\n\nThis is a long document ({num_windows} windows total, "
            f"window {window_idx} of {num_windows}). Be especially selective — "
            f"extract only passages of genuine contemplative depth that could stand "
            f"alongside Ramana Maharshi's direct teachings. For a document this size, "
            f"expect roughly {max_passages} total passages across all windows, so "
            f"most windows should yield 0-1 passages."
        )

    return prompt


def extract_passages(entry: dict, window: dict, model_name: str,
                     num_windows: int = 1, window_idx: int = 1,
                     max_passages: int = 30) -> list[dict] | None:
    """Send a text window to the LLM and parse passage extractions."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                entry, window, num_windows, window_idx, max_passages)},
        ],
        "temperature": 0.15,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(
            f"{VLLM_BASE}/chat/completions",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        choice0 = (data.get("choices") or [{}])[0] or {}
        msg = choice0.get("message") or {}
        content = msg.get("content")
        if content is None:
            # Some providers can return tool-call style responses with null content.
            # Log a small snippet for diagnosis and treat as no extractions.
            print("    LLM returned null message.content; skipping this window")
            try:
                snippet = json.dumps(data)[:500]
            except Exception:
                snippet = str(data)[:500]
            print(f"    Response snippet (first 500): {snippet}")
            return None

        raw = str(content).strip()

        # Strip markdown fencing
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)

        if not isinstance(result, list):
            print(f"    Expected array, got {type(result).__name__}")
            return None

        return result

    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        # raw may not be defined if we failed before reading content
        try:
            print(f"    Raw (first 300): {raw[:300]}")
        except Exception:
            pass
        return None
    except requests.exceptions.RequestException as e:
        print(f"    Request error: {e}")
        return None


def resolve_passage_text(lines: list[str], start_line: int, end_line: int) -> str:
    """Extract passage text from source lines using 1-indexed line numbers."""
    # Clamp to valid range
    s = max(0, start_line - 1)
    e = min(len(lines), end_line)
    return "".join(lines[s:e]).strip()


def deduplicate_passages(passages: list[dict]) -> list[dict]:
    """
    Remove overlapping passages from different windows.
    If two passages overlap in line range, keep the one with higher confidence
    (or the first one if tied).
    """
    if not passages:
        return passages

    # Sort by start_line
    passages.sort(key=lambda p: p.get("start_line", 0))

    kept = [passages[0]]
    for p in passages[1:]:
        prev = kept[-1]
        # Check overlap
        if p.get("start_line", 0) <= prev.get("end_line", 0):
            # Overlap — keep the one with higher confidence
            conf_rank = {"high": 2, "medium": 1}
            if conf_rank.get(p.get("standalone_confidence"), 0) > conf_rank.get(prev.get("standalone_confidence"), 0):
                kept[-1] = p
            # Otherwise keep prev (already in list)
        else:
            kept.append(p)

    return kept


def validate_passage(p: dict) -> list[str]:
    """Light validation of a passage record."""
    warnings = []

    valid_themes = {
        "self-inquiry", "ego-dissolution", "true-nature", "witness-awareness",
        "surrender", "effortless-effort", "beyond-knowledge", "direct-experience",
        "nature-of-mind", "stillness", "attention", "detachment",
        "illusion", "unity", "ordinary-life", "impermanence",
        "suffering", "desire", "liberation", "death",
        "guru-grace", "devotion", "compassion", "the-source", "presence",
        "the-sacred", "practice-as-life", "paradox", "body",
        "scripture-and-tradition",
    }

    for th in p.get("themes", []):
        if th not in valid_themes:
            warnings.append(f"invalid theme: {th}")

    threads = p.get("semantic_threads", [])
    if not threads:
        warnings.append("no semantic_threads")
    elif len(threads) > 3:
        warnings.append(f"too many semantic_threads: {len(threads)}")

    if p.get("start_line", 0) >= p.get("end_line", 0):
        warnings.append(f"bad line range: {p.get('start_line')}-{p.get('end_line')}")

    text = p.get("text", "")
    word_count = len(text.split())
    if word_count < 20:
        warnings.append(f"very short passage: {word_count} words")
    elif word_count > 800:
        warnings.append(f"very long passage: {word_count} words")

    return warnings


VALID_CATEGORIES = ("essays", "religion", "philosophy", "poetry", "other", "nature")

# ── Output paths ───────────────────────────────────────────────────────
PASSAGES_DIR = Path("filtered_guten/passages")
RUNS_DIR = PASSAGES_DIR / "runs"
CORPUS_FILE = PASSAGES_DIR / "corpus.jsonl"


def load_existing_doc_ids() -> set[str]:
    """Load doc_ids already present in passages/corpus.jsonl."""
    ids = set()
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    doc_id = rec.get("doc_id")
                    if doc_id:
                        ids.add(doc_id)
                except json.JSONDecodeError:
                    continue
    return ids


def rebuild_corpus():
    """Rebuild corpus.jsonl by merging all run files, deduplicating by passage_id."""
    seen = set()
    records = []
    if not RUNS_DIR.exists():
        return 0
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


def main():
    parser = argparse.ArgumentParser(description="Extract contemplative passages")
    parser.add_argument(
        "--annotations", default="filtered_guten/annotations_sample.jsonl",
        help="Path to annotations JSONL",
    )
    parser.add_argument(
        "--corpus-dir", default="filtered_guten",
        help="Base directory containing corpus text files",
    )
    parser.add_argument(
        "--category", default=None, choices=VALID_CATEGORIES,
        help="Process only texts from this category (essays, religion, philosophy, poetry, other, nature)",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Retries per window on failure",
    )
    parser.add_argument(
        "--window-tokens", type=int, default=6000,
        help="Window size in approximate tokens",
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=1500,
        help="Overlap between windows in approximate tokens",
    )
    parser.add_argument(
        "--max-texts", type=int, default=None,
        help="Process at most N texts (for testing)",
    )
    parser.add_argument(
        "--max-passages-per-doc", type=int, default=30,
        help="Advisory max passages per document (calibrates LLM for long docs)",
    )
    args = parser.parse_args()

    # Load qualified entries
    ann_path = Path(args.annotations)
    if not ann_path.exists():
        print(f"Annotations not found: {ann_path}")
        sys.exit(1)

    entries = load_qualified_entries(ann_path)
    print(f"Qualified entries (core/useful + high/medium): {len(entries)}")

    if not entries:
        print("No entries qualify. Check annotations file.")
        sys.exit(1)

    # Filter by category if specified
    if args.category:
        prefix = args.category + "/"
        entries = [e for e in entries if e.get("path", "").startswith(prefix)]
        print(f"Filtered to category '{args.category}': {len(entries)} entries")
        if not entries:
            print(f"No entries in category '{args.category}'.")
            sys.exit(1)

    # Skip texts already in corpus.jsonl
    existing_doc_ids = load_existing_doc_ids()
    if existing_doc_ids:
        before = len(entries)
        entries = [e for e in entries if e.get("id") not in existing_doc_ids]
        skipped = before - len(entries)
        print(f"Skipping {skipped} texts already in {CORPUS_FILE} "
              f"({len(existing_doc_ids)} doc_ids in corpus)")
        if not entries:
            print("All texts already processed. Nothing to do.")
            sys.exit(0)

    if args.max_texts:
        entries = entries[:args.max_texts]
        print(f"Limited to {len(entries)} texts for testing")

    print(f"\nTexts to process: {len(entries)}")

    # Connect
    model_name = get_model_name()

    corpus_dir = Path(args.corpus_dir)

    # Prepare output: timestamped run file
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_file = RUNS_DIR / f"{timestamp}.jsonl"
    print(f"Run output: {run_file}")

    total_passages = 0
    total_warnings = 0
    total_windows = 0

    with open(run_file, "w") as out:
        for doc_idx, entry in enumerate(entries):
            doc_id = entry.get("id", "?")
            doc_path = entry.get("path", "")
            title = entry.get("title", "?")[:60]
            author = entry.get("author", "?")

            print(f"\n[{doc_idx+1}/{len(entries)}] {doc_id} — {author}: {title}")
            print(f"  relevance={entry.get('corpus_relevance')}, "
                  f"depth={entry.get('contemplative_depth')}, "
                  f"orientation={entry.get('orientation')}")

            # Locate text file
            text_path = corpus_dir / doc_path
            if not text_path.exists():
                print(f"  TEXT NOT FOUND: {text_path}")
                continue

            lines = load_text_with_lines(text_path)
            print(f"  {len(lines)} lines loaded")

            # Create windows
            windows = make_windows(
                lines,
                window_tokens=args.window_tokens,
                overlap_tokens=args.overlap_tokens,
            )
            print(f"  {len(windows)} windows")

            # Process each window
            doc_passages = []
            for w_idx, window in enumerate(windows):
                total_windows += 1
                w_label = f"  window {w_idx+1}/{len(windows)} (lines {window['start_line']}-{window['end_line']})"

                result = None
                for attempt in range(1 + args.retries):
                    if attempt > 0:
                        print(f"    retry {attempt}...")
                        time.sleep(2)
                    result = extract_passages(
                        entry, window, model_name,
                        num_windows=len(windows),
                        window_idx=w_idx + 1,
                        max_passages=args.max_passages_per_doc,
                    )
                    if result is not None:
                        break

                if result is None:
                    print(f"{w_label}: FAILED")
                    continue

                if len(result) == 0:
                    # No passages in this window — that's fine
                    continue

                # Resolve actual text for each passage
                for p in result:
                    try:
                        start = int(p.get("start_line", 0))
                        end = int(p.get("end_line", 0))
                    except (ValueError, TypeError):
                        print(f"    Warning: Invalid line numbers in passage: start_line={p.get('start_line')}, end_line={p.get('end_line')}")
                        continue
                    p["text"] = resolve_passage_text(lines, start, end)

                print(f"{w_label}: {len(result)} passages found")
                doc_passages.extend(result)

            # Deduplicate across overlapping windows
            doc_passages = deduplicate_passages(doc_passages)

            # Write passages with inherited metadata
            for p_idx, passage in enumerate(doc_passages):
                passage_record = {
                    "passage_id": f"{doc_id}-{p_idx+1:03d}",
                    "doc_id": doc_id,
                    "author": entry.get("author"),
                    "title": entry.get("title"),
                    "tradition": entry.get("tradition"),
                    "tradition_secondary": entry.get("tradition_secondary"),
                    "era": entry.get("era"),
                    "original_language": entry.get("original_language"),
                    "orientation": entry.get("orientation"),
                    "doc_contemplative_depth": entry.get("contemplative_depth"),
                    "doc_themes": entry.get("themes"),
                    "accessibility": entry.get("accessibility"),

                    "start_line": passage.get("start_line"),
                    "end_line": passage.get("end_line"),
                    "passage_type": passage.get("passage_type"),
                    "themes": passage.get("themes", []),
                    "semantic_threads": passage.get("semantic_threads", []),
                    "standalone_confidence": passage.get("standalone_confidence"),
                    "notes": passage.get("notes"),

                    "text": passage.get("text", ""),
                    "text_words": len(passage.get("text", "").split()),
                }

                # Validate
                warnings = validate_passage(passage_record)
                if warnings:
                    passage_record["_warnings"] = warnings
                    total_warnings += 1

                out.write(json.dumps(passage_record) + "\n")
                total_passages += 1

            print(f"  → {len(doc_passages)} passages extracted (after dedup)")

    # Rebuild merged corpus from all run files
    corpus_total = rebuild_corpus()

    # Summary
    print("\n" + "=" * 60)
    print(f"Done.")
    print(f"Run file: {run_file}")
    print(f"Texts processed: {len(entries)}")
    print(f"Windows processed: {total_windows}")
    print(f"Passages extracted (this run): {total_passages}")
    print(f"Passages with warnings: {total_warnings}")
    print(f"Corpus total: {corpus_total} passages in {CORPUS_FILE}")


if __name__ == "__main__":
    main()