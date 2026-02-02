#!/usr/bin/env python3
"""
SFT Training for Contemplative Dialogue Model
==============================================

Fine-tunes a merged Phi-2 base model on contemplative Q&A dialogues
using LoRA targeting upper transformer layers.

Usage:
    # Basic usage (train on base model)
    python sft_training.py --model_path /path/to/base/model --data_path /path/to/data.jsonl
    
    # Load LoRA checkpoint, merge, then train
    python sft_training.py --model_path microsoft/phi-2 \
        --checkpoint phi2-contemplative-lora/checkpoint-2250 \
        --data_path /path/to/data.jsonl

Requirements:
    pip install torch transformers trl peft datasets accelerate
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_model_id_for_path(model_path: str) -> str:
    # Hugging Face IDs like "microsoft/phi-2" become a filesystem-friendly name.
    return model_path.replace("/", "_").replace(":", "_")


def _patch_adapter_config_base_model(output_dir: Path, merged_base_model_path: Path) -> int:
    """
    Update PEFT adapter_config.json files under output_dir to point at merged_base_model_path.
    This ensures inference loads the correct merged base before applying the SFT LoRA adapter.
    """
    merged_base_str = str(merged_base_model_path.resolve())
    patched = 0
    for cfg_path in output_dir.rglob("adapter_config.json"):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("base_model_name_or_path") != merged_base_str:
                cfg["base_model_name_or_path"] = merged_base_str
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                patched += 1
        except Exception as e:
            logger.warning(f"Could not patch {cfg_path}: {e}")
    return patched


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def convert_to_chatml(record: dict) -> dict:
    """
    Convert a record from the source format to ChatML-compatible format.
    
    Input format:
        {"id": "...", "messages": [{"role": "human", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Output format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    converted_messages = []
    for msg in record.get("messages", []):
        role = msg["role"]
        # Map "human" -> "user" for ChatML compatibility
        if role == "human":
            role = "user"
        converted_messages.append({
            "role": role,
            "content": msg["content"]
        })
    
    return {"messages": converted_messages}


def prepare_dataset(data_path: str, val_split: float = 0.1) -> tuple[Dataset, Dataset]:
    """
    Load and prepare dataset for SFT.
    
    Returns:
        (train_dataset, val_dataset)
    """
    raw_records = load_jsonl(data_path)
    logger.info(f"Loaded {len(raw_records)} records from {data_path}")
    
    # Convert to ChatML format
    converted = [convert_to_chatml(r) for r in raw_records]
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(converted)
    
    # Split into train/val
    split = dataset.train_test_split(test_size=val_split, seed=42)
    
    logger.info(f"Train: {len(split['train'])} records, Val: {len(split['test'])} records")
    
    return split['train'], split['test']


# =============================================================================
# MODEL SETUP
# =============================================================================

def _load_lora_checkpoint(base_model, checkpoint_dir: Path):
    """
    Load LoRA adapter with key remapping to handle torch.compile artifacts.
    
    Same logic as inference script to ensure compatibility.
    """
    from peft import PeftModel, LoraConfig, get_peft_model
    from safetensors.torch import load_file
    import json
    
    # Load adapter config
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise ValueError(f"Adapter config not found at {config_path}")
    
    with open(config_path) as f:
        adapter_config = json.load(f)
    
    # Create LoRA config and wrap base model
    lora_config = LoraConfig(
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        target_modules=adapter_config["target_modules"],
        lora_dropout=adapter_config.get("lora_dropout", 0.0),
        bias=adapter_config.get("bias", "none"),
        task_type=adapter_config.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(base_model, lora_config)
    
    # Load checkpoint weights
    ckpt_path = checkpoint_dir / "adapter_model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "adapter_model.bin"
        if not ckpt_path.exists():
            raise ValueError(f"Adapter weights not found at {checkpoint_dir}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        state_dict = load_file(str(ckpt_path))
    
    # Remap keys (handle torch.compile artifacts)
    remapped_state = {}
    for key, value in state_dict.items():
        new_key = key
        # Remove _orig_mod. prefix (from torch.compile)
        new_key = new_key.replace("_orig_mod.", "")
        # Ensure base_model. prefix exists (PEFT wraps model as base_model.model)
        if not new_key.startswith("base_model."):
            new_key = "base_model." + new_key
        # Add .default. to lora keys if missing
        if ".lora_A.weight" in new_key and ".default." not in new_key:
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in new_key and ".default." not in new_key:
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        remapped_state[new_key] = value
    
    # Load state dict
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    
    if missing:
        lora_missing = [k for k in missing if "lora" in k]
        if lora_missing:
            logger.warning(f"Missing LoRA keys: {len(lora_missing)} (showing first 3: {lora_missing[:3]})")
    
    return model


def setup_model_and_tokenizer(
    model_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_upper_layers: bool = True,
    num_layers: int = 32,  # Phi-2 has 32 layers
    upper_fraction: float = 1/3,  # Target top third
    lora_checkpoint: Optional[Path] = None,
    merged_model_output: Optional[Path] = None,
):
    """
    Load model and apply LoRA configuration.
    
    Args:
        model_path: Path to base model (or merged model)
        lora_rank: LoRA rank for new LoRA (if not loading checkpoint)
        lora_alpha: LoRA alpha for new LoRA (if not loading checkpoint)
        lora_dropout: Dropout for LoRA layers
        target_upper_layers: If True, only apply LoRA to upper layers
        num_layers: Total number of transformer layers
        upper_fraction: Fraction of upper layers to target
        lora_checkpoint: Optional path to LoRA checkpoint to load and merge
        merged_model_output: Optional path to save merged model (if checkpoint provided)
    
    Returns:
        (model, tokenizer, merged_model_path)
        merged_model_path is None if no checkpoint was merged, otherwise the path where merged model was saved
    """
    logger.info(f"Loading base model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure chat template is set (ChatML)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
        )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Fix Phi-2 config compatibility (pad_token_id may not exist)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
        model_config.pad_token_id = model_config.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id ({model_config.eos_token_id})")
    
    # Try flash_attention_2, fallback to sdpa if not available
    attn_impl = "sdpa"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using flash_attention_2")
    except ImportError:
        logger.info("flash_attention_2 not available, using sdpa")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    
    merged_model_path = None
    
    # If checkpoint provided, load LoRA, merge, and save
    if lora_checkpoint is not None:
        checkpoint_path = Path(lora_checkpoint)
        logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
        
        # Load LoRA adapter
        peft_model = _load_lora_checkpoint(model, checkpoint_path)
        
        # Merge LoRA into base model
        logger.info("Merging LoRA adapter into base model...")
        merged_model = peft_model.merge_and_unload()
        
        # Determine output path for merged model
        if merged_model_output is None:
            # Extract checkpoint name (e.g., "checkpoint-2250" from "phi2-contemplative-lora/checkpoint-2250")
            checkpoint_name = checkpoint_path.name
            merged_model_path = Path(model_path).parent / f"{Path(model_path).stem}_{checkpoint_name}_merged"
        else:
            merged_model_path = Path(merged_model_output)
        
        merged_model_path.mkdir(parents=True, exist_ok=True)
        
        # Save merged model
        logger.info(f"Saving merged model to {merged_model_path}")
        merged_model.save_pretrained(str(merged_model_path))
        tokenizer.save_pretrained(str(merged_model_path))

        # Make future PEFT adapters record the merged model as their base.
        merged_base_str = str(merged_model_path.resolve())
        try:
            merged_model.config._name_or_path = merged_base_str
        except Exception:
            pass
        
        # Use merged model for further training
        model = merged_model
    
    # Calculate which layers to target
    if target_upper_layers:
        num_upper = int(num_layers * upper_fraction)
        start_layer = num_layers - num_upper
        layers_to_transform = list(range(start_layer, num_layers))
        logger.info(f"Targeting upper layers: {layers_to_transform}")
    else:
        layers_to_transform = None
        logger.info("Targeting all layers")
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=layers_to_transform,
    )
    
    # Apply new LoRA on top of (possibly merged) model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer, merged_model_path


# =============================================================================
# EVALUATION CALLBACK (CRITIC INTEGRATION)
# =============================================================================

class CriticEvaluationCallback(TrainerCallback):
    """
    Callback to run critic evaluation at checkpoints.
    
    Integrates with the AlivenessCritic to track response quality
    beyond just loss metrics.
    """
    
    def __init__(
        self,
        eval_prompts: list[str],
        critic=None,  # AlivenessCritic instance
        eval_every_n_steps: Optional[int] = None,
        eval_at_epoch_end: bool = True,
        output_dir: str = "./critic_evals",
    ):
        self.eval_prompts = eval_prompts
        self.critic = critic
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_at_epoch_end = eval_at_epoch_end
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.eval_history = []
    
    def _run_evaluation(self, model, tokenizer, step: int, epoch: float):
        """Generate responses and score with critic."""
        if self.critic is None:
            logger.warning("No critic configured, skipping evaluation")
            return
        
        model.eval()
        scores = []
        results = []
        
        for prompt in self.eval_prompts:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Generate
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Score with critic
            critic_result = self.critic.evaluate(prompt, response)
            scores.append(critic_result.score)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "score": critic_result.score,
                "alive_signals": critic_result.alive_signals,
                "stale_signals": critic_result.stale_signals,
            })
        
        # Compute stats
        mean_score = sum(scores) / len(scores)
        above_threshold = sum(1 for s in scores if s >= 6.0) / len(scores)
        
        eval_record = {
            "step": step,
            "epoch": epoch,
            "mean_score": mean_score,
            "above_threshold_pct": above_threshold,
            "min_score": min(scores),
            "max_score": max(scores),
            "results": results,
        }
        
        self.eval_history.append(eval_record)
        
        # Log
        logger.info(f"Critic eval @ step {step}: mean={mean_score:.2f}, above_threshold={above_threshold:.1%}")
        
        # Save
        eval_path = self.output_dir / f"eval_step_{step}.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_record, f, indent=2)
        
        model.train()
        
        return eval_record
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_every_n_steps and state.global_step % self.eval_every_n_steps == 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model and tokenizer:
                self._run_evaluation(model, tokenizer, state.global_step, state.epoch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_at_epoch_end:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model and tokenizer:
                self._run_evaluation(model, tokenizer, state.global_step, state.epoch)


# =============================================================================
# TRAINING
# =============================================================================

def train(
    model_path: str,
    data_path: str,
    output_dir: str = "./sft_output",
    # LoRA params
    lora_rank: int = 16,
    lora_alpha: int = 32,
    target_upper_layers: bool = True,
    # Checkpoint loading
    lora_checkpoint: Optional[str] = None,
    merged_model_output: Optional[str] = None,
    # Training params
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 2048,
    # Evaluation
    eval_prompts_path: Optional[str] = None,
    critic=None,
):
    """
    Run SFT training.
    
    Args:
        model_path: Path to base model (or merged model)
        data_path: Path to JSONL training data
        output_dir: Where to save checkpoints
        lora_rank: LoRA rank for new LoRA adapter
        lora_alpha: LoRA alpha for new LoRA adapter
        target_upper_layers: Only apply LoRA to upper third of layers
        lora_checkpoint: Optional path to LoRA checkpoint to load and merge before training
        merged_model_output: Optional path to save merged model (default: {model_path}_{checkpoint_name}_merged)
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_seq_length: Maximum sequence length
        eval_prompts_path: Path to file with evaluation prompts (one per line)
        critic: AlivenessCritic instance for evaluation
    """
    
    # Prepare data
    train_dataset, val_dataset = prepare_dataset(data_path)
    
    # Setup model
    checkpoint_path = Path(lora_checkpoint) if lora_checkpoint else None
    merged_output_path = Path(merged_model_output) if merged_model_output else None
    if checkpoint_path is not None and merged_output_path is None:
        merged_output_path = (
            Path(output_dir)
            / "merged_models"
            / f"{_safe_model_id_for_path(model_path)}_{checkpoint_path.name}_merged"
        )
    
    model, tokenizer, merged_path = setup_model_and_tokenizer(
        model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_upper_layers=target_upper_layers,
        lora_checkpoint=checkpoint_path,
        merged_model_output=merged_output_path,
    )
    
    if merged_path:
        logger.info(f"Merged model saved to {merged_path}. Using merged model for SFT training.")
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_length=max_seq_length,  # SFTConfig uses max_length, not max_seq_length
        packing=False,  # Set True if you want sequence packing
        dataset_text_field=None,  # We're using messages format
    )
    
    # Setup callbacks
    callbacks = []
    
    if eval_prompts_path and critic:
        with open(eval_prompts_path, 'r') as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
        
        critic_callback = CriticEvaluationCallback(
            eval_prompts=eval_prompts,
            critic=critic,
            eval_at_epoch_end=True,
            output_dir=f"{output_dir}/critic_evals",
        )
        callbacks.append(critic_callback)
        logger.info(f"Loaded {len(eval_prompts)} evaluation prompts")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Ensure all saved adapter configs under output_dir point at the merged base.
    if merged_path:
        patched = _patch_adapter_config_base_model(Path(output_dir), merged_path)
        logger.info(f"Patched {patched} adapter_config.json files to use merged base: {merged_path.resolve()}")
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT Training for Contemplative Dialogue")
    
    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model (or merged model)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL training data")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./sft_output",
                        help="Output directory for checkpoints")
    
    # LoRA checkpoint loading
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint to load and merge before training")
    parser.add_argument("--merged_model_output", type=str, default=None,
                        help="Path to save merged model (default: {model_path}_{checkpoint_name}_merged)")
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--all_layers", action="store_true",
                        help="Target all layers instead of just upper third")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # Evaluation
    parser.add_argument("--eval_prompts", type=str, default=None,
                        help="Path to evaluation prompts file (one per line)")
    parser.add_argument("--use_critic", action="store_true",
                        help="Use AlivenessCritic for evaluation")
    
    args = parser.parse_args()
    
    # Setup critic if requested
    critic = None
    if args.use_critic:
        try:
            from aliveness_critic import LocalCritic
            critic = LocalCritic(backend="openrouter", model="anthropic/claude-sonnet-4")
            logger.info("Initialized AlivenessCritic")
        except Exception as e:
            logger.warning(f"Could not initialize critic: {e}")
    
    # Run training
    train(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_upper_layers=not args.all_layers,
        lora_checkpoint=args.checkpoint,
        merged_model_output=args.merged_model_output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        eval_prompts_path=args.eval_prompts,
        critic=critic,
    )


if __name__ == "__main__":
    main()
