#!/usr/bin/env python3
"""
SFT Training for Contemplative Dialogue Model
==============================================

Fine-tunes a merged Phi-2 base model on contemplative Q&A dialogues
using LoRA targeting upper transformer layers.

Usage:
    python sft_training.py --model_path /path/to/merged/model --data_path /path/to/data.jsonl

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

def setup_model_and_tokenizer(
    model_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_upper_layers: bool = True,
    num_layers: int = 32,  # Phi-2 has 32 layers
    upper_fraction: float = 1/3,  # Target top third
):
    """
    Load model and apply LoRA configuration.
    
    Args:
        model_path: Path to merged base model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout for LoRA layers
        target_upper_layers: If True, only apply LoRA to upper layers
        num_layers: Total number of transformer layers
        upper_fraction: Fraction of upper layers to target
    
    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    
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
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2",  # Remove if not available
    )
    
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
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


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
        model_path: Path to merged base model
        data_path: Path to JSONL training data
        output_dir: Where to save checkpoints
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        target_upper_layers: Only apply LoRA to upper third of layers
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
    model, tokenizer = setup_model_and_tokenizer(
        model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_upper_layers=target_upper_layers,
    )
    
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
        max_seq_length=max_seq_length,
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
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT Training for Contemplative Dialogue")
    
    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged base model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL training data")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./sft_output",
                        help="Output directory for checkpoints")
    
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
            from aliveness_critic import LLMCritic
            critic = LLMCritic(provider="anthropic", model="claude-sonnet-4-20250514")
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
