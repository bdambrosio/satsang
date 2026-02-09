# In Honor of Bhagavan

A contemplative web application inspired by the teachings of Bhagavan Sri Ramana Maharshi. It generates responses in the spirit of the teachings, displays related passages from world contemplative traditions, and presents Ramana's own words as a daily opening text.

This is not a simulation of Bhagavan, not a substitute for teachers or direct inquiry, and makes no claim to authority. It exists to make these teachings more accessible. See the [About page](/about) for more.

## Web Application

### Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (index.html)                                │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │  Main panel          │  │  Sidebar               │ │
│  │  - Header passage    │  │  - Related passage     │ │
│  │  - Query input       │  │  - "Another voice"     │ │
│  │  - Response          │  │    alternate            │ │
│  │  - Expanded response │  │                        │ │
│  │  - Direct / Conv mode│  │                        │ │
│  └─────────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌──────────────────────────────────────────────────────┐
│  Flask API (ramana_api.py)                           │
│                                                      │
│  /              → Main page                          │
│  /about         → About page                         │
│  /api/nan-yar   → Personalized header passage        │
│  /api/query     → Initial contemplative response     │
│  /api/expand    → Expanded response with depth       │
│  /api/retrieve-passages → Sidebar passage retrieval  │
└──────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────┐
│  LLM Backend    │  │  Passage Corpus (FAISS)        │
│  (local vLLM    │  │  ~18K passages, ~52K semantic   │
│   or OpenRouter) │  │  threads from world traditions  │
└─────────────────┘  └────────────────────────────────┘
```

### Query Flow

1. **Header passage**: On page load, a passage from Nan Yar, Ulladu Narpadu, or Upadesa Undiyar is selected based on the visitor's session history (semantic similarity to prior conversation threads). Bracketed glosses in the text are rendered as hover/tap tooltips.

2. **Initial response** (`/api/query`): The visitor's question is answered by an LLM in a contemplative, non-prescriptive register. An aliveness critic scores the response; below-threshold responses are regenerated or returned as silence.

3. **Expanded response** (`/api/expand`): A deeper elaboration drawing on Ramana's Commentaries and related teachings. Filtered by a coherence check before display.

4. **Sidebar passage** (`/api/retrieve-passages`): A passage from the cross-tradition corpus that illuminates the same experiential territory from a different angle. Retrieved via:
   - LLM query rewrite into semantic threads (same embedding space as corpus)
   - FAISS similarity search over ~52K thread vectors
   - Depth-weighted and author-frequency-dampened scoring
   - LLM post-filter for coherence and contemplative register
   - Up to 16 candidates retrieved, filtered in two rounds of 8
   - An alternate passage (preferring a different tradition/author) is provided for the "Another voice" button

### Two Interaction Modes

- **Direct**: Single response + expandable depth. Terse, contemplative pointing.
- **Conversational**: Scrolling dialogue with expand-on-click. Gentler, more interactive.

### Running

#### With Docker (OpenRouter backend)

```bash
docker build -t satsang .
docker run -p 5001:5001 \
  -e OPENROUTER_API_KEY=your-key-here \
  -v $(pwd)/sessions:/app/sessions \
  satsang
```

#### Local development (local vLLM backend)

```bash
pip install -r requirements-runtime.txt
python ramana_api.py \
  --llm-backend local \
  --llm-url http://localhost:5000/v1 \
  --port 5001
```

### Source Texts

All texts are drawn from public domain sources. Header passages come from:
- *Nan Yar? (Who Am I?)*
- *Ulladu Narpadu (Forty Verses on Reality)*
- *Upadesa Undiyar (Thirty Verses on Instruction)*

The sidebar corpus (~18K passages) spans Advaita Vedanta, Buddhism, Sufism, Christian mysticism, Taoism, Stoicism, Neoplatonism, and other contemplative traditions, sourced from Project Gutenberg.

### Project Structure

```
satsang/
├── ramana_api.py                        # Flask API server
├── Dockerfile                           # Container build
├── requirements-runtime.txt             # Runtime dependencies
├── templates/
│   ├── index.html                       # Main page
│   └── about.html                       # About page
├── static/
│   └── style.css                        # Styles
├── src/
│   ├── aliveness_critic.py              # Response generation + critic
│   ├── contemplative_rag.py             # RAG for expanded responses
│   ├── filtered_passages_rag.py         # Sidebar passage retrieval
│   └── ...                              # Training scripts, parsers
├── ramana/                              # Ramana source texts
├── filtered_guten/
│   └── filtered_passages/
│       └── corpus.jsonl                 # Sidebar passage corpus
└── sessions/                            # Session data (runtime)
```

## Training Pipeline

The contemplative response model is fine-tuned from Microsoft's Phi-2 using a two-stage approach:

1. **Continued pretraining** on a curated non-conversational contemplative subset of the Gutenberg corpus (~100M tokens) via LoRA
2. **Supervised fine-tuning** on contemplative dialogues, primarily from *Talks with Sri Ramana Maharshi*

The goal is to resist the "closure reflex" -- the habitual move toward resolution, helpfulness, and answer-giving that kills contemplative space. The model should point rather than explain, leave space for inquiry, and respond to what's beneath the surface question.

See `src/continued-pretrain-lora.py`, `src/sft_training.py`, and `src/parse-talks.py` for the training pipeline.

## Feedback

This project welcomes feedback, especially from scholars and practitioners of Bhagavan's teachings. Contact: bruce@tuuyi.com

## License

See LICENSE file for details.
