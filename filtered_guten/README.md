# filtered_guten — Cross-Tradition Contemplative Corpus

This directory contains the processing pipeline for extracting contemplative
passages from Project Gutenberg texts. The goal is to build a cross-tradition
corpus of self-contained passages that can appear alongside Ramana Maharshi's
teachings on the satsang website — sidebar "resonances" from other traditions
that illuminate the same experiential territory.

## What's in the corpus

The source material is a curated subset of Project Gutenberg texts spanning
multiple contemplative traditions: Advaita Vedanta, Buddhism (Theravada,
Mahayana, Zen, Tibetan), Hindu devotional and yoga literature, Sufism,
Christian mysticism, Taoism, Stoicism, Neoplatonism, Transcendentalism, and
others.

Texts are organized into category subdirectories (`essays/`, `religion/`,
`philosophy/`, `poetry/`, `nature/`, `other/`) and catalogued in `manifest.json`.

### Sample manifest entry

```json
{
  "id": "PG10001",
  "title": "Apocolocyntosis",
  "author": "Seneca, Lucius Annaeus",
  "category": "other",
  "reason": "priority-author",
  "tokens": 7209,
  "chars": 30556,
  "path": "other/PG10001_Seneca_Lucius_Annaeus_Apocolocyntosis.txt"
}
```

Fields:
- **id**: Project Gutenberg ID
- **title / author**: From PG metadata
- **category**: Broad tradition grouping (directory name)
- **reason**: Why this text was included (e.g. `priority-author`, `keyword-match`)
- **tokens / chars**: Approximate size
- **path**: Relative path to the plain-text file within `filtered_guten/`

> **Note:** The manifest, text files, and category subdirectories are not
> committed to git (see `.gitignore`). Only the pipeline scripts and this
> README are tracked.

## Directory structure

The pipeline produces two main output directories:

- **`passages/`** — Output from Stage 3 (passage identification)
  - `corpus.jsonl` — Merged corpus of all extracted passages
  - `runs/` — Timestamped run files (e.g. `20260206T143000Z.jsonl`)

- **`filtered_passages/`** — Output from Stage 4 (filtering)
  - `corpus.jsonl` — Final filtered corpus (used by the web app)
  - `runs/` — Timestamped run files (e.g. `20260206T143000Z.jsonl`)

Both directories support incremental runs: each stage skips work already present
in its `corpus.jsonl` file, allowing safe re-runs and incremental processing.

## Processing pipeline

The pipeline has four stages, each implemented as a standalone script. All
LLM-dependent stages require a vLLM server running on `http://0.0.0.0:5000/v1`.

### Stage 1: Annotate — `annotate.py`

Enriches manifest entries with LLM-generated metadata.

```
python annotate.py --manifest filtered_guten/manifest.json --n 100 --output annotations_sample.jsonl
```

**What it does:**
- Randomly samples `--n` entries from the manifest
- Sends each entry to the LLM, which produces structured metadata based on its
  knowledge of the text (author + title)
- Copies `category`, `path`, and `tokens` directly from the manifest to ensure
  they're always correct regardless of LLM output
- Validates the LLM response against controlled vocabularies
- Outputs one JSONL record per entry

**Key output fields added by the LLM:**
- `tradition` / `tradition_secondary` — e.g. `stoic`, `buddhist-zen`
- `orientation` — literary form: `dialogue`, `poetry`, `prose-treatise`, etc.
- `contemplative_depth` — 1 (adjacent) to 3 (direct pointing)
- `themes` — controlled vocabulary of 30 contemplative themes
- `era` — `ancient`, `classical`, `medieval`, `early-modern`, `modern`, `contemporary`
- `corpus_relevance` — `core`, `useful`, `marginal`, `questionable`
- `confidence` — `high`, `medium`, `low`

### Stage 2: Statistics — `stats.py`

Quick summary of annotation results.

```
python stats.py annotations_sample.jsonl
```

Prints cross-tabulation of `corpus_relevance` x `confidence` with token counts.
Useful for deciding whether to re-run annotation with different sampling or to
adjust the filtering thresholds in later stages.

### Stage 3: Passage Identification — `passage_identification.py`

Extracts self-contained contemplative passages from qualified texts.

```
python passage_identification.py \
  --annotations filtered_guten/annotations-full.jsonl \
  --corpus-dir filtered_guten \
  --max-passages-per-doc 25

# Process only a specific category
python passage_identification.py \
  --annotations filtered_guten/annotations-full.jsonl \
  --corpus-dir filtered_guten \
  --category religion \
  --max-passages-per-doc 25
```

**What it does:**
- Filters annotations to `core`/`useful` relevance + `high`/`medium` confidence
- Optionally filters by `--category` (essays, religion, philosophy, poetry, other, nature)
- Skips `doc_id`s already present in `passages/corpus.jsonl` (incremental runs)
- For each qualifying text, loads the full plain-text file
- Slides overlapping windows (default 6000 tokens, 1500 overlap) through the text
- Sends each window to the LLM with document metadata and asks it to identify
  self-contained passages with exact line numbers
- For long documents (>20 windows), injects an advisory calibration message
  telling the LLM to be more selective (`--max-passages-per-doc`)
- Resolves extracted line ranges back to actual text
- Deduplicates overlapping passages across window boundaries

**Output structure:**
- Each run writes to a timestamped file: `passages/runs/YYYYMMDDTHHMMSSZ.jsonl`
- After each run, rebuilds `passages/corpus.jsonl` by merging all run files
- Output is flushed after each document (crash-safe)

**Output fields per passage:**
- `passage_id` — e.g. `PG10001-001`
- `doc_id`, `author`, `title`, `tradition`, `era`, `orientation`, etc. (inherited)
- `start_line`, `end_line` — line range in the source text
- `passage_type` — `dialogue`, `verse`, `prose`, `aphorism`, `prayer`, `scripture`
- `themes` — controlled vocabulary
- `semantic_threads` — 1-3 natural-language descriptions of contemplative meaning
  (used as embedding keys for retrieval)
- `standalone_confidence` — `high` or `medium`
- `text` — the extracted passage text
- `text_words` — word count

### Stage 4: Filter Passages — `filter_passages.py`

Two-pass quality filter on extracted passages. Produces the final filtered
corpus ready for ingest by the web app.

```
# Uses passages/corpus.jsonl as input by default
python filter_passages.py

# Customize thresholds
python filter_passages.py --reject 0.35 --accept 0.55

# Skip LLM pass (embedding-only)
python filter_passages.py --no-llm

# Override input file
python filter_passages.py --passages custom_input.jsonl
```

**Input:**
- Default: reads from `passages/corpus.jsonl` (output from Stage 3)
- Can override with `--passages` to process a different file

**Pass 1 — Embedding distance (cheap, deterministic):**
- Encodes each passage's `semantic_threads` with a sentence transformer
- Compares against 30 theme glosses (detailed descriptions of each contemplative theme)
- Passages below `--reject` threshold (default 0.35) are dropped immediately
- Passages above `--accept` threshold (default 0.55) are kept immediately
- Everything in between is borderline

**Pass 2 — LLM judgment (borderline only):**
- Sends the full passage text to the LLM for a yes/no contemplative relevance call
- Returns `relevant: true/false` with `confidence` and `reason`

**Incremental output:**
- Reads `filtered_passages/corpus.jsonl` (if it exists) to build a set of
  already-filtered `passage_id`s; those passages are skipped
- Each run writes accepted passages to a timestamped file under
  `filtered_passages/runs/` (e.g. `20260206T143000Z.jsonl`)
- After the run, rebuilds `filtered_passages/corpus.jsonl` by merging and
  deduplicating all run files — this is the single file the web app reads

## Dependencies

- Python 3.11+
- `requests` — HTTP client for vLLM API
- `sentence-transformers` — embedding model for theme filtering (Stage 4)
- `numpy` — used by theme filtering

All LLM stages require a vLLM server (or compatible OpenAI API) at
`http://0.0.0.0:5000/v1`. The model is auto-detected from the `/models`
endpoint.

## Controlled vocabulary: themes

The same 30-theme vocabulary is used across annotation, passage identification,
and theme filtering:

`self-inquiry`, `ego-dissolution`, `true-nature`, `witness-awareness`,
`surrender`, `effortless-effort`, `beyond-knowledge`, `direct-experience`,
`nature-of-mind`, `stillness`, `attention`, `detachment`, `illusion`, `unity`,
`ordinary-life`, `impermanence`, `suffering`, `desire`, `liberation`, `death`,
`guru-grace`, `devotion`, `compassion`, `the-source`, `presence`,
`the-sacred`, `practice-as-life`, `paradox`, `body`, `scripture-and-tradition`

## Remaining work

- **Scale annotation**: Current runs sample a subset (`--n`). Full corpus
  annotation requires multiple runs or increasing `--n` to cover all entries.
- **Tune thresholds**: The embedding reject/accept thresholds in `filter_passages.py`
  and the `--max-passages-per-doc` value may need adjustment as more data flows
  through the pipeline.
- **Integration**: Accepted passages are loaded into the retrieval system via
  `FilteredPassagesRAG` (see `src/filtered_passages_rag.py`), which builds a FAISS
  index from `semantic_threads` for sidebar passage retrieval.
- **Human review**: Low-confidence annotations and borderline LLM judgments
  should be spot-checked.
- **Deduplication across documents**: Some passages (e.g. the same Upanishad
  verse in multiple compilations) may appear in different source texts.
