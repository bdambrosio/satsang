#!/usr/bin/env python3
"""
Continued Pretraining for Phi-2 on Contemplative Corpus
========================================================

This script performs domain-adaptive continued pretraining on Phi-2
using a filtered corpus of non-instrumental text (poetry, philosophy,
religion, essays, nature writing).

The goal is to shift the model's priors toward contemplative language
before any instruction fine-tuning.

Usage:
    # First, prepare the data
    python continued_pretrain.py prepare --corpus-dir ./filtered_guten --output-dir ./prepared_data
    
    # Then train
    python continued_pretrain.py train --data-dir ./prepared_data --output-dir ./phi2-contemplative

    # Or do both
    python continued_pretrain.py all --corpus-dir ./filtered_guten --output-dir ./phi2-contemplative

Requirements:
    pip install torch transformers datasets accelerate wandb
    
Hardware:
    - Tested on RTX 6000 Pro (96GB) — can run full precision
    - For smaller GPUs, enable gradient checkpointing and reduce batch size
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_NAME = "microsoft/phi-2"

# Training hyperparameters (tuned for RTX 6000 Pro 96GB)
DEFAULT_CONFIG = {
    # Learning rate: much lower than initial pretraining to avoid forgetting
    # Phi-2 was trained with peak LR ~1e-4, we use ~1e-5
    "learning_rate": 1e-5,
    
    # Batch size (tuned for 96GB VRAM)
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,  # Effective batch = 32
    
    # Sequence length (Phi-2 supports 2048)
    "max_seq_length": 2048,
    
    # Training duration
    # For ~300M tokens with effective batch 32 and seq_len 2048:
    # Steps = 300M / (32 * 2048) ≈ 4500 steps
    # We'll do 1-2 epochs through the data
    "num_train_epochs": 2,
    
    # Optimizer
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Memory optimization
    "gradient_checkpointing": False,  # Not needed with 96GB
    "bf16": True,  # Use bf16 on Ampere+/Blackwell
    "tf32": True,  # Enable TF32 for faster compute
    
    # Logging
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    
    # Misc
    "seed": 42,
    "dataloader_num_workers": 8,  # You have cores and RAM
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_dataset(
    corpus_dir: Path,
    output_dir: Path,
    max_seq_length: int = 2048,
    test_split: float = 0.02,
):
    """
    Prepare the corpus for training.
    
    Tokenizes files incrementally to avoid memory blowup.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import gc
    
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Phi-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Collect all text files
    logger.info(f"Collecting text files from {corpus_dir}")
    text_files = sorted(corpus_dir.rglob("*.txt"))
    logger.info(f"Found {len(text_files)} text files")
    
    # Process incrementally: tokenize files in batches, accumulate token IDs
    BATCH_SIZE = 100  # Files per batch
    
    all_token_ids = []
    total_chars = 0
    
    logger.info("Tokenizing files in batches...")
    
    for batch_start in tqdm(range(0, len(text_files), BATCH_SIZE), desc="Tokenizing"):
        batch_files = text_files[batch_start:batch_start + BATCH_SIZE]
        batch_texts = []
        
        for txt_file in batch_files:
            try:
                content = txt_file.read_text(encoding='utf-8', errors='replace').strip()
                if content:
                    batch_texts.append(content)
                    total_chars += len(content)
            except Exception as e:
                logger.warning(f"Error reading {txt_file}: {e}")
        
        if not batch_texts:
            continue
        
        # Tokenize this batch (join with separator)
        batch_text = "\n\n".join(batch_texts)
        
        tokenized = tokenizer(
            batch_text,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
        )
        
        all_token_ids.extend(tokenized["input_ids"])
        
        # Free memory
        del batch_texts, batch_text, tokenized
        gc.collect()
    
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Total tokens: {len(all_token_ids):,}")
    
    # Chunk into sequences
    logger.info(f"Chunking into sequences of {max_seq_length} tokens")
    
    chunks = []
    for i in tqdm(range(0, len(all_token_ids) - max_seq_length + 1, max_seq_length), 
                  desc="Chunking"):
        chunk = all_token_ids[i:i + max_seq_length]
        chunks.append({"input_ids": chunk, "labels": chunk.copy()})
    
    # Handle remainder
    remainder = len(all_token_ids) % max_seq_length
    if remainder > max_seq_length // 2:
        last_chunk = all_token_ids[-max_seq_length:]
        if len(last_chunk) < max_seq_length:
            padding = [tokenizer.pad_token_id] * (max_seq_length - len(last_chunk))
            last_chunk = padding + last_chunk
        chunks.append({"input_ids": last_chunk, "labels": last_chunk.copy()})
    
    # Free the big list
    del all_token_ids
    gc.collect()
    
    logger.info(f"Created {len(chunks)} sequences")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = Dataset.from_list(chunks)
    
    # Free chunks list
    del chunks
    gc.collect()
    
    # Split
    split = dataset.train_test_split(test_size=test_split, seed=42)
    
    logger.info(f"Train: {len(split['train'])} sequences")
    logger.info(f"Eval: {len(split['test'])} sequences")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(output_dir))
    
    # Save tokenizer for later
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    
    logger.info(f"Dataset saved to {output_dir}")
    
    return split


# =============================================================================
# TRAINING
# =============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    resume_from: Optional[Path] = None,
    use_wandb: bool = False,
):
    """
    Run continued pretraining.
    """
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Enable TF32 for faster training on Ampere+
    if config["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(str(data_dir))
    
    # Load tokenizer
    tokenizer_path = data_dir / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model {MODEL_NAME}")
    
    # Load config first and fix missing pad_token_id (Phi-2 compatibility with newer transformers)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
        model_config.pad_token_id = model_config.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id ({model_config.eos_token_id})")
    
    # Load model - let transformers pick the best available attention
    model_kwargs = {
        "config": model_config,
        "torch_dtype": torch.bfloat16 if config["bf16"] else torch.float32,  # Deprecation warning is harmless
        "trust_remote_code": True,
        "attn_implementation": "sdpa",  # Force standard attention - most compatible
    }
    
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    logger.info("Model loaded with spda attention")
    
    # Enable gradient checkpointing if requested
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Data collator (handles padding/batching)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        #overwrite_output_dir=True,
        
        # Batch size
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        
        # Training duration
        num_train_epochs=config["num_train_epochs"],
        
        # Optimizer
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        
        # Precision
        bf16=config["bf16"],
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=config["logging_steps"],
        report_to="wandb" if use_wandb else "none",
        
        # Saving
        save_steps=config["save_steps"],
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        
        # Misc
        seed=config["seed"],
        dataloader_num_workers=config["dataloader_num_workers"],
        
        # Resume
        resume_from_checkpoint=str(resume_from) if resume_from else None,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    # Train!
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("Training complete!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continued pretraining for Phi-2 on contemplative corpus"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prep_parser.add_argument("--corpus-dir", type=Path, required=True,
                            help="Directory containing text files")
    prep_parser.add_argument("--output-dir", type=Path, default=Path("./prepared_data"),
                            help="Where to save prepared dataset")
    prep_parser.add_argument("--max-seq-length", type=int, default=2048)
    prep_parser.add_argument("--test-split", type=float, default=0.02)
    
    # Train command  
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--data-dir", type=Path, required=True,
                             help="Directory with prepared dataset")
    train_parser.add_argument("--output-dir", type=Path, required=True,
                             help="Where to save trained model")
    train_parser.add_argument("--resume-from", type=Path, default=None,
                             help="Resume from checkpoint")
    train_parser.add_argument("--wandb", action="store_true",
                             help="Enable Weights & Biases logging")
    train_parser.add_argument("--lr", type=float, default=None,
                             help="Override learning rate")
    train_parser.add_argument("--epochs", type=int, default=None,
                             help="Override number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=None,
                             help="Override per-device batch size")
    
    # All-in-one command
    all_parser = subparsers.add_parser("all", help="Prepare and train")
    all_parser.add_argument("--corpus-dir", type=Path, required=True)
    all_parser.add_argument("--output-dir", type=Path, required=True)
    all_parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_dataset(
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            test_split=args.test_split,
        )
    
    elif args.command == "train":
        config_overrides = {}
        if args.lr:
            config_overrides["learning_rate"] = args.lr
        if args.epochs:
            config_overrides["num_train_epochs"] = args.epochs
        if args.batch_size:
            config_overrides["per_device_train_batch_size"] = args.batch_size
        
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config_overrides if config_overrides else None,
            resume_from=args.resume_from,
            use_wandb=args.wandb,
        )
    
    elif args.command == "all":
        data_dir = args.output_dir / "prepared_data"
        prepare_dataset(
            corpus_dir=args.corpus_dir,
            output_dir=data_dir,
        )
        train(
            data_dir=data_dir,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
        )


if __name__ == "__main__":
    main()