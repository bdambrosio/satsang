# Satsang: A Non-Conversational Contemplative Language Model

This project fine-tunes Microsoft's Phi-2 model to generate contemplative responses that resist the "closure reflex" — the habitual move toward resolution, helpfulness, and answer-giving that kills contemplative space.

## The Problem: Why "Non-Conversational"?

Most language models are trained to be helpful assistants. They:
- Provide answers and solutions
- Use prescriptive language ("you should", "try this")
- Offer closure and resolution
- Mirror the questioner's framing
- Perform wisdom rather than embodying it

**This project aims to create a model that does the opposite:**
- Leaves space for the questioner's own discovery
- Turns energy back as genuine inquiry (not rhetorical deflection)
- Refuses to close what wants to stay open
- Is comfortable with not-knowing
- Responds to what's beneath the surface question
- Uses minimal words—no more than the moment requires
- Points rather than explains

The model is trained on contemplative dialogues (primarily from Sri Ramana Maharshi) where responses create space rather than fill it.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Data Preparation                                         │
│     ├─ Parse Q / A record (parse-talks.py)                 │
│     ├─ Filter corpus (filter_spgc_corpus.py)                │
│        (a selected set of non-conversational meditative texts from gutenberg.org. |
|     └─ Prepare training data (prepared_data/)              │
│                                                               │
│  2. Continued Pretraining                                    │
│     └─ LoRA fine-tuning on non-conversational meditative corpus           │
│        (continued-pretrain-lora.py)                         │
│                                                               │
│  3. Supervised Fine-Tuning (Optional)                       │
│     └─ SFT on Q&A dialogues (sft_training.py)              │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Inference Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [User Query]                                                │
│       ↓                                                       │
│  [Fine-tuned Model] → [Candidate Response]                   │
│       ↓                                                       │
│  [Aliveness Critic] → [Score < threshold?]                   │
│       ↓                    ↓                                  │
│  [Return Response]    [Regenerate / Silence]                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Development & Execution Flow

### 1. Data Preparation

#### Parse Contemplative Texts
Extract Q&A dialogues from source texts (e.g., "Talks with Sri Ramana Maharshi"):

```bash
python parse-talks.py ramana/Talks-with-Sri-Ramana-Maharshi--complete.txt \
    --output ramana/Talks-parsed.jsonl
```

This extracts structured conversations in JSONL format:
```json
{
  "id": "talk_001",
  "messages": [
    {"role": "human", "content": "What is the Self?"},
    {"role": "assistant", "content": "Who is asking?"}
  ]
}
```

#### Filter Corpus (Optional)
Filter large text corpora for contemplative/spiritual content:

```bash
python filter_spgc_corpus.py --input filtered_guten/ --output prepared_data/
```

### 2. Continued Pretraining with LoRA

Fine-tune Phi-2 on the contemplative corpus using Low-Rank Adaptation (LoRA):

```bash
python continued-pretrain-lora.py train \
    --data-dir ./prepared_data \
    --output-dir ./phi2-contemplative-lora \
    --wandb
```

**Key Configuration:**
- **LoRA Rank**: 64 (moderate expressiveness)
- **LoRA Alpha**: 128 (scaling factor = 2.0)
- **Target Modules**: Auto-discovered (q_proj, k_proj, v_proj, dense, fc1, fc2)
- **Training**: 3 epochs, batch size 64, learning rate 2e-4

**Why LoRA?**
The hypothesis: shifting from "helpful assistant" to "contemplative interlocutor" is a low-rank transformation that can be captured without full weight updates. This allows:
- Efficient training (only ~3% of parameters updated)
- Preservation of base model capabilities
- Faster iteration cycles

**Checkpoints:**
- Saved every 1000 steps
- Checkpoint format: `phi2-contemplative-lora/checkpoint-{step}/`
- Each checkpoint contains LoRA adapter weights (`adapter_model.safetensors`)

### 3. Supervised Fine-Tuning (Optional)

Further fine-tune on structured Q&A dialogues:

```bash
python sft_training.py \
    --model_path ./phi2-contemplative-lora/checkpoint-1000 \
    --data_path ramana/Talks-parsed.jsonl \
    --output_dir ./sft_output
```

This uses TRL's `SFTTrainer` for supervised fine-tuning on dialogue pairs.

### 4. Inference

#### Basic Inference
Load a checkpoint and generate responses:

```bash
python phi2-contemplative-inference-lora.py \
    --checkpoint phi2-contemplative-lora/checkpoint-1000 \
    --query "I feel lost in my practice."
```

**Key Features:**
- Automatic LoRA adapter loading with key remapping (handles `torch.compile` artifacts)
- Configurable generation parameters (temperature, top_p, max_tokens)
- Support for 4-bit/8-bit quantization for memory efficiency

#### With Aliveness Critic
Use the critic to filter stale responses:

```python
from aliveness_critic import AlivenessCritic, ContemplativeGenerator

critic = AlivenessCritic(model="openai/gpt-4", threshold=6.0)
generator = ContemplativeGenerator(
    generator_url="http://localhost:5000/v1",
    critic=critic,
    max_attempts=3
)

response = generator.generate("I feel lost in my practice.")
```

The critic evaluates responses on a 0-10 scale:
- **9-10**: Genuinely alive, creates space, masterful
- **7-8**: Good, mostly alive with minor staleness
- **5-6**: Mixed, some alive qualities but closure reflex present
- **3-4**: Mostly stale, performing rather than being
- **1-2**: Generic spiritual chatbot output
- **0**: Actively harmful to contemplative space

If a response scores below the threshold, the generator automatically regenerates with higher temperature. After `max_attempts`, it may return silence rather than a stale response.

## Key Components

### `continued-pretrain-lora.py`
LoRA continued pretraining script. Handles:
- Auto-discovery of target modules for Phi-2
- Gradient checkpointing for memory efficiency
- `torch.compile` for faster training (with key remapping for checkpoints)
- WandB integration for experiment tracking

### `sft_training.py`
Supervised fine-tuning on dialogue pairs. Uses TRL's `SFTTrainer` for efficient training.

### `phi2-contemplative-inference-lora.py`
Inference script with:
- Custom LoRA loading that handles `torch.compile` key prefixes (`_orig_mod.`)
- Automatic `.default.` adapter name insertion for PEFT compatibility
- Support for merging LoRA weights into base model for faster inference

### `aliveness_critic.py`
Runtime scaffolding system that evaluates responses for "aliveness":
- **LLMCritic**: Uses OpenAI/Anthropic APIs for evaluation
- **LocalCritic**: Uses local model API endpoints
- **ContemplativeGenerator**: Wraps generator with critic filtering

### `parse-talks.py`
Parser for extracting Q&A dialogues from unstructured text. Handles:
- Multiple speaker markers (M.:, D.:, Maharshi:, Devotee:, etc.)
- Inline dialogues on same line
- Embedded questions in narrative
- Normalization of various formatting artifacts

### `diagnose-lora-checkpoint.py`
Diagnostic tool for inspecting LoRA checkpoint structure:
- Key naming patterns
- Weight statistics
- Compatibility checks

## Example Outputs

**Query**: "I feel lost in my practice."

**Base Phi-2** (conversational):
> "I understand that feeling lost in your practice can be challenging. Here are some strategies that might help: 1) Set clear goals, 2) Find a mentor, 3) Break down your practice into smaller steps..."

**Fine-tuned Model** (contemplative):
> "Who is it that feels lost?"

The fine-tuned model turns the question back, creating space for inquiry rather than providing solutions.

## Requirements

See `requirements.txt` for full dependencies. Core requirements:

```bash
pip install torch transformers peft datasets accelerate trl safetensors
```

Optional (for aliveness critic):
```bash
pip install openai anthropic httpx wandb
```

## Project Structure

```
satsang/
├── continued-pretrain-lora.py      # LoRA pretraining
├── sft_training.py                  # Supervised fine-tuning
├── phi2-contemplative-inference-lora.py  # Inference
├── aliveness_critic.py              # Response filtering
├── parse-talks.py                   # Data extraction
├── filter_spgc_corpus.py            # Corpus filtering
├── diagnose-lora-checkpoint.py      # Checkpoint diagnostics
├── requirements.txt                 # Dependencies
├── prepared_data/                   # Training data (gitignored)
├── phi2-contemplative-lora/         # LoRA checkpoints (gitignored)
└── ramana/                          # Source texts (gitignored)
```

## Notes

- **Checkpoint Compatibility**: LoRA checkpoints saved with `torch.compile` include `_orig_mod.` prefixes. The inference script automatically remaps these keys for PEFT compatibility.
- **Training Progress**: Early checkpoints (e.g., step 1000) show the model learning the *form* (contemplative voice) before the *content* (wisdom). Full training (3 epochs, ~7100 steps) is needed for substantial content shift.
- **Memory**: LoRA training is memory-efficient (~5-6GB GPU memory for Phi-2 with batch size 64), but full model inference may require quantization for smaller GPUs.

## License

See LICENSE file for details.
