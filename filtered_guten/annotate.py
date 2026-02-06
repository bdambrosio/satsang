#!/usr/bin/env python3
"""
Annotate contemplative corpus manifest entries using vLLM-served LLM.
Randomly samples entries, sends them one-at-a-time, outputs JSONL.
"""

import json
import random
import sys
import time
import argparse
import requests
from pathlib import Path

VLLM_BASE = "http://0.0.0.0:5000/v1"

SYSTEM_PROMPT = """You are an expert annotator for a contemplative literature corpus. Your task is to enrich metadata for texts drawn from Project Gutenberg, which will be used in a retrieval system serving a website devoted to the teachings of Ramana Maharshi. The cross-tradition corpus provides sidebar resonances — passages from other traditions that illuminate the same experiential territory Ramana points to.

You will be given a manifest entry containing: id, title, author, category, reason, tokens, chars, path.

Based on your knowledge of the text (from title and author), produce enriched metadata. If you are not confident you know the text well enough to annotate accurately, set "confidence" to "low" and do your best — these will be reviewed separately.

Respond with a single JSON object on one line. No markdown, no explanation, no preamble, no trailing text. Fields:

{
  "id": "<preserved from input>",
  "title": "<preserved from input>",
  "author": "<preserved from input>",
  "category": "<preserved from input>",
  "path": "<preserved from input>",
  "tokens": <preserved from input>,
  "tradition": "<primary tradition>",
  "tradition_secondary": "<secondary tradition or null>",
  "orientation": "<primary literary form>",
  "contemplative_depth": <1-3>,
  "themes": ["<theme1>", "<theme2>", ...],
  "era": "<period>",
  "original_language": "<language>",
  "accessibility": <1-3>,
  "confidence": "<high|medium|low>",
  "corpus_relevance": "<core|useful|marginal|questionable>",
  "notes": "<brief free-text if anything needs flagging, else null>"
}

FIELD DEFINITIONS AND ALLOWED VALUES:

tradition (required, single value):
advaita-vedanta, buddhist-theravada, buddhist-mahayana, buddhist-zen, buddhist-tibetan, buddhist-general, hindu-devotional, hindu-yoga, hindu-general, sufi, islamic-general, christian-mystical, christian-general, jewish-mystical, jewish-general, taoist, stoic, neoplatonist, transcendentalist, existentialist, western-philosophical, classical-greek, roman, hermetic, indigenous, secular-contemplative, mixed-or-unclear

tradition_secondary: same values as tradition, or null.

orientation (required, single value):
dialogue, poetry, prose-treatise, prose-commentary, devotional-prayer, autobiography-confession, aphorism-collection, narrative, letters, sermons-talks, scripture, mixed

contemplative_depth (required, integer 1-3):
3 = Direct pointing at inner experience (Eckhart's sermons, Upanishads, Rumi's mystical poetry)
2 = Reflective philosophical engagement with contemplative themes (Marcus Aurelius, Emerson)
1 = Adjacent — primarily ethical/cosmological/literary but with contemplative passages

themes (required, array of 0-5 strings, ordered by relevance):
Assign ONLY themes that are genuinely central to the text. Fewer is better — a text that is only marginally contemplative may deserve 0-1 themes. Do not stretch to fill a quota. If no themes apply, use an empty array []:
Controlled vocabulary:
- self-inquiry: investigating "Who am I?", turning attention to the subject
- ego-dissolution: unreality or death of the separate self
- true-nature: what one already is; the Self, Buddha-nature, divine ground
- witness-awareness: pure consciousness distinct from its contents
- surrender: letting go of personal will, yielding to what is
- effortless-effort: paradox that practice aims at what already is
- beyond-knowledge: limits of intellect, apophatic approaches, unknowing
- direct-experience: primacy of immediate knowing over doctrine
- nature-of-mind: how thought arises, mechanics of mental activity
- stillness: silence, cessation, the ground beneath mental noise
- attention: focused awareness as practice, vigilance, watchfulness
- detachment: non-clinging, dispassion, freedom from outcomes
- illusion: the world as not what it seems; maya, the veil, the dream
- unity: nonduality, collapse of subject-object, all is one
- ordinary-life: the sacred in the everyday, no need for renunciation
- impermanence: transience of phenomena, what passes vs what remains
- suffering: nature and role of pain, dissatisfaction as doorway
- desire: craving, attachment, mechanism of ego-sustenance
- liberation: moksha, nirvana, freedom; present recognition not future state
- death: physical death, ego-death, death as spiritual catalyst
- guru-grace: the teacher's role, transmission, grace
- devotion: bhakti, love of God, the heart's movement toward the divine
- compassion: response to suffering from understanding
- the-source: origin of manifestation, the Heart, Brahman, Tao, Godhead
- presence: being itself prior to doing or knowing, pure "I am"
- the-sacred: encounter with the numinous, awe, the holy
- practice-as-life: integration of spiritual work into all activity
- paradox: teaching through contradiction, logic that breaks logic
- body: embodiment, breath, somatic awareness in spiritual life
- scripture-and-tradition: relationship to inherited teaching, map vs territory

era (required, single value):
ancient (before 200 CE), classical (200-800 CE), medieval (800-1400 CE), early-modern (1400-1700 CE), modern (1700-1900 CE), contemporary (1900+)

original_language (required, single value):
sanskrit, pali, classical-chinese, persian, arabic, greek, latin, english, german, french, italian, spanish, portuguese, japanese, tibetan, hebrew, other

accessibility (required, integer 1-3):
3 = Immediately engaging, clear language, accessible to anyone
2 = Requires some familiarity or patience but rewarding
1 = Dense, technical, or archaic; primarily for scholars

confidence (required): high, medium, or low

corpus_relevance (required, single value):
How useful this text is likely to be for a contemplative retrieval corpus:
- core: Deeply contemplative; central to the corpus purpose
- useful: Contains significant contemplative passages or themes, even if not the primary focus
- marginal: Tangentially related; may contain occasional relevant passages but is not primarily contemplative
- questionable: Unclear why this text is in the corpus; may not belong

notes: brief free-text or null. Flag: partial texts, compilations, misleading titles, translation concerns, etc."""


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


def annotate_entry(entry: dict, model_name: str) -> dict | None:
    """Send a single manifest entry to the LLM and parse the response."""
    user_msg = f"Annotate the following manifest entry:\n\n{json.dumps(entry, indent=2)}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            f"{VLLM_BASE}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fencing if the model wraps it
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        
        # Copy category, path, and tokens from manifest entry to ensure they're preserved
        # (LLM might not always include them correctly)
        if "category" in entry:
            result["category"] = entry["category"]
        if "path" in entry:
            result["path"] = entry["path"]
        if "tokens" in entry:
            result["tokens"] = entry["tokens"]
        
        return result

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw response: {raw[:500]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Request error: {e}")
        return None


def validate_annotation(ann: dict) -> list[str]:
    """Light validation — returns list of warnings."""
    warnings = []

    valid_traditions = {
        "advaita-vedanta", "buddhist-theravada", "buddhist-mahayana",
        "buddhist-zen", "buddhist-tibetan", "buddhist-general",
        "hindu-devotional", "hindu-yoga", "hindu-general", "sufi",
        "islamic-general", "christian-mystical", "christian-general",
        "jewish-mystical", "jewish-general", "taoist", "stoic",
        "neoplatonist", "transcendentalist", "existentialist",
        "western-philosophical", "classical-greek", "roman", "hermetic",
        "indigenous", "secular-contemplative", "mixed-or-unclear",
    }
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
    valid_orientations = {
        "dialogue", "poetry", "prose-treatise", "prose-commentary",
        "devotional-prayer", "autobiography-confession", "aphorism-collection",
        "narrative", "letters", "sermons-talks", "scripture", "mixed",
    }
    valid_eras = {
        "ancient", "classical", "medieval", "early-modern", "modern",
        "contemporary",
    }
    valid_confidence = {"high", "medium", "low"}

    t = ann.get("tradition")
    if t not in valid_traditions:
        warnings.append(f"invalid tradition: {t}")

    ts = ann.get("tradition_secondary")
    if ts is not None and ts not in valid_traditions:
        warnings.append(f"invalid tradition_secondary: {ts}")

    o = ann.get("orientation")
    if o not in valid_orientations:
        warnings.append(f"invalid orientation: {o}")

    cd = ann.get("contemplative_depth")
    if cd not in (1, 2, 3):
        warnings.append(f"invalid contemplative_depth: {cd}")

    themes = ann.get("themes", [])
    if not (2 <= len(themes) <= 5):
        warnings.append(f"theme count {len(themes)}, expected 2-5")
    for th in themes:
        if th not in valid_themes:
            warnings.append(f"invalid theme: {th}")

    era = ann.get("era")
    if era not in valid_eras:
        warnings.append(f"invalid era: {era}")

    acc = ann.get("accessibility")
    if acc not in (1, 2, 3):
        warnings.append(f"invalid accessibility: {acc}")

    conf = ann.get("confidence")
    if conf not in valid_confidence:
        warnings.append(f"invalid confidence: {conf}")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Annotate manifest entries")
    parser.add_argument(
        "--manifest", default="filtered_guten/manifest.json",
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of entries to sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output", default="annotations_sample.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Retries per entry on failure",
    )
    args = parser.parse_args()

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Handle both list and dict-with-list formats
    if isinstance(manifest, dict):
        # Try common keys
        for key in ("texts", "entries", "items", "books"):
            if key in manifest:
                manifest = manifest[key]
                break
        else:
            print(f"Manifest is a dict with keys: {list(manifest.keys())}")
            print("Expected a list or a dict containing a list. Please check format.")
            sys.exit(1)

    print(f"Loaded {len(manifest)} entries from manifest")

    # Sample
    random.seed(args.seed)
    sample = random.sample(manifest, min(args.n, len(manifest)))
    print(f"Sampled {len(sample)} entries (seed={args.seed})")

    # Connect to vLLM
    model_name = get_model_name()

    # Process
    output_path = Path(args.output)
    stats = {"success": 0, "failed": 0, "warnings": 0}
    confidence_counts = {"high": 0, "medium": 0, "low": 0}

    with open(output_path, "w") as out:
        for i, entry in enumerate(sample):
            label = f"[{i+1}/{len(sample)}] {entry.get('id', '?')} — {entry.get('author', '?')}: {entry.get('title', '?')[:60]}"
            print(label)

            result = None
            for attempt in range(1 + args.retries):
                if attempt > 0:
                    print(f"  retry {attempt}...")
                    time.sleep(2)
                result = annotate_entry(entry, model_name)
                if result is not None:
                    break

            if result is None:
                print(f"  FAILED after {1 + args.retries} attempts")
                stats["failed"] += 1
                # Write a failure record so we know what was skipped
                fail_record = {"id": entry.get("id"), "_error": "annotation_failed"}
                out.write(json.dumps(fail_record) + "\n")
                continue

            # Validate
            warnings = validate_annotation(result)
            if warnings:
                print(f"  WARNINGS: {'; '.join(warnings)}")
                result["_warnings"] = warnings
                stats["warnings"] += 1

            conf = result.get("confidence", "?")
            if conf in confidence_counts:
                confidence_counts[conf] += 1
            print(f"  ✓ confidence={conf}, themes={result.get('themes', [])}")

            out.write(json.dumps(result) + "\n")
            stats["success"] += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Done. Output: {output_path}")
    print(f"Success: {stats['success']}, Failed: {stats['failed']}, With warnings: {stats['warnings']}")
    print(f"Confidence: {confidence_counts}")


if __name__ == "__main__":
    main()