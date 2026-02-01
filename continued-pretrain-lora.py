#!/usr/bin/env python3
"""
LoRA Continued Pretraining for Phi-2 on Contemplative Corpus
=============================================================

Low-rank adaptation variant. The hypothesis: shifting from 
"helpful assistant" to "contemplative interlocutor" is a 
low-rank transformation that can be captured without full 
weight updates.

Uses same prepared data as continued_pretrain.py.

Usage:
    python continued_pretrain_lora.py train \
        --data-dir ./prepared_data \
        --output-dir ./phi2-contemplative-lora

Requirements:
    pip install torch transformers datasets accelerate peft wandb
"""

import argparse
import logging
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

MODEL_NAME = "microsoft/phi-2"

# LoRA configuration
LORA_CONFIG = {
    # Rank: higher = more expressiveness, more params
    # r=64 is moderate; try r=128 or r=256 if underfitting
    "r": 64,
    
    # Alpha: scaling factor. alpha/r is the actual scaling.
    # Common practice: alpha = r or alpha = 2*r
    "lora_alpha": 128,
    
    # Dropout for regularization
    "lora_dropout": 0.05,
    
    # Target modules - will be auto-discovered for Phi-2
    # Common patterns: "q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"
    # OR: "Wqkv", "out_proj", "mlp.fc1", "mlp.fc2" (depending on implementation)
    # Set to None to auto-discover
    "target_modules": None,
    
    # Train biases too
    "bias": "none",
    
    # Task type
    "task_type": "CAUSAL_LM",
}

# Training hyperparameters - optimized for large GPU (96GB Blackwell)
DEFAULT_CONFIG = {
    # Learning rate: can be higher with LoRA (only updating small adapter)
    "learning_rate": 2e-4,
    
    # Batch size - increased for large GPU with LoRA
    # Effective batch = per_device * gradient_accumulation * num_devices
    "per_device_train_batch_size": 64,  # Increased from 4
    "gradient_accumulation_steps": 1,  # Reduced from 8 (effective batch = 64)
    
    # More epochs since we're learning less per step
    "num_train_epochs": 3,
    
    # Optimizer
    # Note: Trainer uses AdamW by default. For fused optimizer (faster on modern GPUs),
    # you'd need to override Trainer.create_optimizer() or use a custom Trainer class.
    # Consider: fused AdamW (via apex or PyTorch 2.0+), Lion, or Adafactor for large batches.
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Precision
    "bf16": True,
    "tf32": True,
    
    # Gradient checkpointing for memory efficiency and longer context
    "gradient_checkpointing": True,
    
    # Logging - less frequent to reduce overhead
    "logging_steps": 50,  # Increased from 10
    "save_steps": 250,  # Increased from 500
    "eval_steps": 250,  # Increased from 500
    
    # Data loading optimizations
    "dataloader_num_workers": 8,  # Benchmark 4/8/16 for your system
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": True,
    
    # Misc
    "seed": 42,
    
    # Attention implementation - try "flash_attention_2" if available
    "attn_implementation": "sdpa",  # or "flash_attention_2"
    
    # Compile model for speed (5-20% speedup)
    "torch_compile": True,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_peft_with_key_remap(base_model, peft_config, checkpoint_dir: Path):
    """
    Load PEFT adapter with key remapping to handle torch.compile artifacts.
    
    The checkpoint may have keys like:
        _orig_mod.base_model.model.model.layers.0.mlp.fc1.lora_A.weight
    
    But PEFT expects:
        base_model.model.model.layers.0.mlp.fc1.lora_A.default.weight
    """
    from peft import get_peft_model
    from safetensors.torch import load_file
    
    # Create fresh PEFT model
    model = get_peft_model(base_model, peft_config)
    
    # Load checkpoint weights
    ckpt_path = checkpoint_dir / "adapter_model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "adapter_model.bin"
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        state_dict = load_file(str(ckpt_path))
    
    # Remap keys
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
    
    # Debug: show sample keys
    sample_keys = list(remapped_state.keys())[:2]
    logger.info(f"Remapped checkpoint keys (sample): {sample_keys}")
    
    # Get expected keys from model
    model_state = model.state_dict()
    expected_keys = [k for k in model_state.keys() if "lora_" in k][:2]
    logger.info(f"Model expects keys (sample): {expected_keys}")
    
    # Load with strict=False to allow partial loading
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    
    # Filter to only LoRA-related missing keys
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        logger.warning(f"Missing LoRA keys after remap: {len(lora_missing)} keys")
        logger.warning(f"Sample missing: {lora_missing[:3]}")
    else:
        logger.info("All LoRA weights loaded successfully")
    
    return model


def discover_target_modules_deprecated(model):
    """
    Discover actual module names in Phi-2 for LoRA targeting.
    
    Phi-2 module names vary by implementation. This function finds
    attention and MLP layers that can be targeted.
    """
    target_modules = []
    module_names = [name for name, _ in model.named_modules()]
    
    # Common attention patterns
    attention_patterns = ["q_proj", "k_proj", "v_proj", "dense", "out_proj", "Wqkv"]
    
    # Common MLP patterns
    mlp_patterns = ["fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
    
    # Find matching modules
    for name in module_names:
        # Skip embedding and output layers
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        
        # Check attention patterns
        for pattern in attention_patterns:
            if pattern in name:
                target_modules.append(name)
                break
        
        # Check MLP patterns
        for pattern in mlp_patterns:
            if pattern in name:
                target_modules.append(name)
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_modules = []
    for mod in target_modules:
        if mod not in seen:
            seen.add(mod)
            unique_modules.append(mod)
    
    return unique_modules


def discover_target_modules(model):
    """Return unique module suffixes for LoRA targeting."""
    suffixes = set()
    
    attention_patterns = ["q_proj", "k_proj", "v_proj", "dense", "out_proj", "Wqkv"]
    mlp_patterns = ["fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
    all_patterns = attention_patterns + mlp_patterns
    
    for name, module in model.named_modules():
        if "embed" in name.lower() or "lm_head" in name.lower():
            continue
        
        # Get the last part of the name (the suffix)
        suffix = name.split(".")[-1]
        
        if suffix in all_patterns:
            suffixes.add(suffix)
    
    return list(suffixes)
    
def verify_lora_targets(model, target_modules):
    """
    Verify that LoRA adapters were actually attached to target modules.
    
    Returns (matched_count, total_targets, matched_modules)
    """
    peft_modules = []
    for name, module in model.named_modules():
        if "lora" in name.lower():
            # Extract base module name from LoRA adapter name
            # e.g., "base_model.model.layers.0.self_attn.q_proj.lora_A" -> "q_proj"
            parts = name.split(".")
            for part in parts:
                if part in target_modules:
                    peft_modules.append(part)
                    break
    
    matched = set(peft_modules)
    targets = set(target_modules)
    matched_count = len(matched & targets)
    
    return matched_count, len(targets), list(matched)


def log_memory_usage(stage: str):
    """
    Log detailed CUDA memory usage for diagnostics.
    
    Helps identify:
    - Allocated vs reserved memory (PyTorch caching)
    - Peak memory usage
    - Memory fragmentation
    """
    if not torch.cuda.is_available():
        logger.info(f"[{stage}] CUDA not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    logger.info(f"[{stage}] CUDA Memory:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved:  {reserved:.2f} GB")
    logger.info(f"  Max allocated: {max_allocated:.2f} GB")
    logger.info(f"  Gap (caching): {reserved - allocated:.2f} GB")
    
    # Reset peak stats for next measurement
    torch.cuda.reset_peak_memory_stats()


def analyze_sequence_lengths(dataset, tokenizer, sample_size: int = 100):
    """
    Analyze sequence lengths in dataset to diagnose memory usage.
    
    Memory scales with batch × seq_len, so long sequences explain high VRAM.
    Handles both raw text and pre-tokenized datasets.
    """
    lengths = []
    sample_indices = min(sample_size, len(dataset))
    
    for i in range(sample_indices):
        example = dataset[i]
        
        # Check if dataset is already tokenized (has input_ids)
        if "input_ids" in example:
            if isinstance(example["input_ids"], list):
                lengths.append(len(example["input_ids"]))
            elif hasattr(example["input_ids"], "__len__"):
                lengths.append(len(example["input_ids"]))
            continue
        
        # Check if dataset has text field
        text = example.get("text", "")
        if isinstance(text, str) and text:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            lengths.append(len(tokens))
        elif isinstance(text, list):
            # Already tokenized as list
            lengths.append(len(text))
        
        # Check labels field (might indicate sequence length)
        if "labels" in example and not lengths:
            if isinstance(example["labels"], list):
                # Count non-padding tokens (assuming -100 is padding)
                labels = example["labels"]
                seq_len = sum(1 for x in labels if x != -100)
                if seq_len > 0:
                    lengths.append(seq_len)
    
    if not lengths:
        logger.warning("Could not analyze sequence lengths - dataset format may be unexpected")
        logger.info(f"Sample dataset keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'empty'}")
        return None
    
    lengths.sort()
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / len(lengths),
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
        "p99": lengths[int(len(lengths) * 0.99)],
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    lora_config: Optional[dict] = None,
    resume_from: Optional[Path] = None,
    use_wandb: bool = False,
    merge_and_save: bool = False,
):
    """
    Run LoRA continued pretraining.
    """
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    
    config = {**DEFAULT_CONFIG, **(config or {})}
    lora_cfg = {**LORA_CONFIG, **(lora_config or {})}
    
    # Enable TF32
    if config["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(str(data_dir))
    
    # Verify dataset splits
    if "train" not in dataset:
        raise ValueError(f"Dataset must contain 'train' split. Found splits: {list(dataset.keys())}")
    
    # Shuffle training dataset to ensure diverse batches
    # Important if corpus is segmented by type (poetry, philosophy, etc.)
    # This ensures batches contain mixed types rather than homogeneous segments
    logger.info(f"Shuffling training dataset (seed={config['seed']}) to ensure diverse batches...")
    train_dataset = dataset["train"].shuffle(seed=config["seed"])
    logger.info(f"Training dataset size: {len(train_dataset)} examples")
    
    # Make eval optional - check if test split exists
    eval_dataset = None
    if "test" in dataset:
        eval_dataset = dataset["test"]
        logger.info(f"Using 'test' split for evaluation ({len(eval_dataset)} examples)")
    elif "validation" in dataset or "val" in dataset:
        eval_dataset = dataset.get("validation") or dataset.get("val")
        logger.info(f"Using 'validation' split for evaluation ({len(eval_dataset)} examples)")
    else:
        logger.warning("No evaluation split found. Training without evaluation.")
    
    # Load tokenizer
    tokenizer_path = data_dir / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set pad_token to eos_token for causal LM
    # NOTE: This means padding tokens will be trained as EOS tokens.
    # This is fine for packed sequences but can inject bias if using lots of padding.
    # Consider packing sequences to fixed length to minimize padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Analyze sequence lengths - critical for memory diagnosis
    # Must be done after tokenizer is loaded
    # Use original dataset before shuffling for consistent analysis
    logger.info("Analyzing sequence lengths in training dataset...")
    seq_stats = None
    try:
        seq_stats = analyze_sequence_lengths(train_dataset, tokenizer, sample_size=500)
        if seq_stats:
            logger.info(f"Sequence length statistics:")
            logger.info(f"  Min: {seq_stats['min']}, Max: {seq_stats['max']}")
            logger.info(f"  Mean: {seq_stats['mean']:.1f}, Median: {seq_stats['median']}")
            logger.info(f"  P95: {seq_stats['p95']}, P99: {seq_stats['p99']}")
            
            # Warn if sequences are very long
            if seq_stats['p95'] > 2048:
                logger.warning(
                    f"⚠️  Long sequences detected (P95={seq_stats['p95']}). "
                    "Memory usage scales with batch × seq_len. "
                    "Consider packing sequences or reducing max_length."
                )
            if seq_stats['max'] > 4096:
                logger.warning(
                    f"⚠️  Very long sequences detected (max={seq_stats['max']}). "
                    "This will cause high memory usage even with small batches."
                )
    except Exception as e:
        logger.warning(f"Could not analyze sequence lengths: {e}")
    
    # Load model
    logger.info(f"Loading model {MODEL_NAME}")
    
    # Fix Phi-2 config compatibility
    model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
        model_config.pad_token_id = model_config.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id ({model_config.eos_token_id})")
    
    # Attention implementation - try flash_attention_2 if available
    attn_impl = config.get("attn_implementation", "sdpa")
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn
            logger.info("Using flash_attention_2")
        except ImportError:
            logger.warning("flash_attention_2 requested but not available. Falling back to sdpa.")
            attn_impl = "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=model_config,
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    logger.info(f"Base model parameters: {model.num_parameters():,}")
    
    # Log memory after base model load
    log_memory_usage("After base model load")
    
    # Discover or use provided target modules
    if lora_cfg["target_modules"] is None:
        logger.info("Auto-discovering target modules...")
        discovered_modules = discover_target_modules(model)
        logger.info(f"Discovered modules: {discovered_modules}")
        target_modules = discovered_modules
    else:
        target_modules = lora_cfg["target_modules"]
        logger.info(f"Using provided target modules: {target_modules}")
    
    # Apply LoRA - different path for fresh start vs resume
    logger.info(f"Applying LoRA with r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=target_modules,
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    if resume_from is not None:
        # Resuming: Load adapter with key remapping to handle torch.compile artifacts
        checkpoint_path = Path(resume_from)
        adapter_path = checkpoint_path / "adapter_model.safetensors"
        if adapter_path.exists() or (checkpoint_path / "adapter_config.json").exists():
            logger.info(f"Resuming from PEFT checkpoint: {resume_from}")
            model = _load_peft_with_key_remap(model, peft_config, checkpoint_path)
        else:
            logger.warning(f"Checkpoint {resume_from} doesn't appear to be a PEFT checkpoint. Starting fresh.")
            model = get_peft_model(model, peft_config)
    else:
        # Fresh start: just apply LoRA config
        model = get_peft_model(model, peft_config)
    
    # Verify LoRA targets were actually matched
    matched_count, total_targets, matched_modules = verify_lora_targets(model, target_modules)
    logger.info(f"LoRA target verification: {matched_count}/{total_targets} modules matched")
    logger.info(f"Matched modules: {matched_modules}")
    
    if matched_count == 0:
        raise RuntimeError(
            "CRITICAL: No LoRA adapters were attached! "
            "Target modules don't match actual model architecture. "
            f"Expected: {target_modules}, Matched: {matched_modules}"
        )
    elif matched_count < total_targets:
        logger.warning(
            f"Only {matched_count}/{total_targets} target modules matched. "
            "Some adapters may not be attached. Check module names."
        )
    
    trainable_params, total_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / total_params
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    # CRITICAL: Verify LoRA is actually active
    if trainable_pct > 5.0:
        logger.error(
            f"⚠️  CRITICAL: {trainable_pct:.2f}% parameters are trainable! "
            "Expected <1% for LoRA. LoRA adapters may not be attached correctly. "
            "This would explain excessive VRAM usage (full fine-tuning instead of LoRA)."
        )
    elif trainable_pct < 0.1:
        logger.info(f"✓ LoRA active: {trainable_pct:.2f}% trainable (expected for LoRA)")
    else:
        logger.warning(
            f"⚠️  {trainable_pct:.2f}% trainable - higher than expected for LoRA. "
            "Verify LoRA adapters are correctly attached."
        )
    
    # Print detailed trainable parameter breakdown
    model.print_trainable_parameters()
    
    # Log memory after LoRA application
    log_memory_usage("After LoRA application")
    
    # Compile model for speed (if enabled)
    if config.get("torch_compile", False):
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    # Data collator
    # NOTE: DataCollatorForLanguageModeling does dynamic padding, which causes
    # variable-length batches and reduces GPU utilization. For better throughput:
    # - Pre-tokenize and pack sequences to fixed length (e.g., 2048 tokens)
    # - Use static padding with a custom collator
    # - This reduces runtime Python overhead and improves GPU utilization
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Verify gradient checkpointing is enabled
    use_grad_checkpointing = config.get("gradient_checkpointing", False)
    if use_grad_checkpointing:
        logger.info("✓ Gradient checkpointing enabled (reduces activation memory by 40-60%)")
    else:
        logger.warning(
            "⚠️  Gradient checkpointing DISABLED. "
            "This means PyTorch stores all activations for backward pass. "
            "Enable gradient_checkpointing=True to reduce memory usage significantly."
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        
        num_train_epochs=config["num_train_epochs"],
        
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        
        bf16=config["bf16"],
        
        gradient_checkpointing=use_grad_checkpointing,
        
        logging_dir=str(output_dir / "logs"),
        logging_steps=config["logging_steps"],
        report_to="wandb" if use_wandb else "none",
        
        save_steps=config["save_steps"],
        save_total_limit=3,
        
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config["eval_steps"] if eval_dataset is not None else None,
        
        seed=config["seed"],
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=config.get("dataloader_pin_memory", False),
        dataloader_persistent_workers=config.get("dataloader_persistent_workers", False),
        
        # Shuffle dataloader (default is True, but making explicit)
        # Trainer shuffles at start of each epoch to ensure diverse batches
        dataloader_drop_last=False,  # Keep last incomplete batch
        
        # Disable column removal - dataset already has correct format (input_ids, labels)
        remove_unused_columns=False,
        
        # Note: resume_from_checkpoint should only be passed to trainer.train(), not here
    )
    
    # Trainer
    # Note: Trainer shuffles dataset at start of each epoch by default
    # We've already shuffled above to ensure initial diversity
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Log training configuration summary
    effective_batch = (
        config["per_device_train_batch_size"] 
        * config["gradient_accumulation_steps"]
        * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    logger.info("=" * 60)
    logger.info("Training Configuration Summary:")
    logger.info(f"  Per-device batch size: {config['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {effective_batch}")
    logger.info(f"  Gradient checkpointing: {use_grad_checkpointing}")
    logger.info(f"  Trainable params: {trainable_pct:.2f}%")
    if seq_stats:
        logger.info(f"  Sequence length (P95): {seq_stats['p95']}")
        logger.info(f"  Sequence length (max): {seq_stats['max']}")
    logger.info("=" * 60)
    
    # Log memory before training starts
    log_memory_usage("Before training start")
    
    # Train
    logger.info("Starting LoRA training...")
    
    # For PEFT checkpoints, adapter weights are already loaded above via _load_peft_with_key_remap.
    # We need to restore optimizer/scheduler state manually since Trainer's resume_from_checkpoint
    # expects full model checkpoints (with model.safetensors.index.json).
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        trainer_state_path = checkpoint_path / "trainer_state.json"
        
        if trainer_state_path.exists():
            import json
            with open(trainer_state_path) as f:
                saved_state = json.load(f)
            
            resume_step = saved_state.get("global_step", 0)
            resume_epoch = saved_state.get("epoch", 0)
            logger.info(f"Resuming from step {resume_step}, epoch {resume_epoch:.2f}")
            
            # Create optimizer/scheduler state restore callback
            class RestoreOptimizerCallback(TrainerCallback):
                def __init__(self, trainer_ref, checkpoint_path, resume_step, resume_epoch):
                    self.trainer_ref = trainer_ref
                    self.checkpoint_path = checkpoint_path
                    self.resume_step = resume_step
                    self.resume_epoch = resume_epoch
                    self.restored = False
                
                def on_train_begin(self, args, state, control, **kwargs):
                    if self.restored:
                        return
                    
                    # Set the starting step/epoch
                    state.global_step = self.resume_step
                    state.epoch = self.resume_epoch
                    
                    # Load optimizer state (trainer.optimizer exists at this point)
                    optimizer_path = self.checkpoint_path / "optimizer.pt"
                    if optimizer_path.exists() and self.trainer_ref.optimizer is not None:
                        logger.info("Loading optimizer state...")
                        optimizer_state = torch.load(optimizer_path, map_location="cpu")
                        self.trainer_ref.optimizer.load_state_dict(optimizer_state)
                    
                    # Load scheduler state
                    scheduler_path = self.checkpoint_path / "scheduler.pt"
                    if scheduler_path.exists() and self.trainer_ref.lr_scheduler is not None:
                        logger.info("Loading scheduler state...")
                        scheduler_state = torch.load(scheduler_path, map_location="cpu")
                        self.trainer_ref.lr_scheduler.load_state_dict(scheduler_state)
                    
                    self.restored = True
                    logger.info(f"Optimizer and scheduler state restored, resuming from step {self.resume_step}")
            
            trainer.add_callback(RestoreOptimizerCallback(trainer, checkpoint_path, resume_step, resume_epoch))
        
        trainer.train()
    else:
        trainer.train()
    
    # Log peak memory after training
    log_memory_usage("After training (peak)")
    
    # Save adapter
    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Optionally merge and save full model
    if merge_and_save:
        logger.info("Merging adapter into base model...")
        logger.warning(
            "Merging forces full fp16/bf16 weights in memory. "
            "Peak memory will spike during merge operation."
        )
        merged_model = model.merge_and_unload()
        merged_path = output_dir / "merged"
        merged_path.mkdir(exist_ok=True)
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        logger.info(f"Merged model saved to {merged_path}")
    
    logger.info("Training complete!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA continued pretraining for Phi-2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run LoRA training")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--resume-from", type=Path, default=None)
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--merge", action="store_true",
                             help="Merge adapter into base model after training")
    
    # Hyperparameters
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    
    # LoRA params
    train_parser.add_argument("--lora-r", type=int, default=None,
                             help="LoRA rank (default: 64)")
    train_parser.add_argument("--lora-alpha", type=int, default=None,
                             help="LoRA alpha (default: 128)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        config_overrides = {}
        if args.lr:
            config_overrides["learning_rate"] = args.lr
        if args.epochs:
            config_overrides["num_train_epochs"] = args.epochs
        if args.batch_size:
            config_overrides["per_device_train_batch_size"] = args.batch_size
        
        lora_overrides = {}
        if args.lora_r:
            lora_overrides["r"] = args.lora_r
        if args.lora_alpha:
            lora_overrides["lora_alpha"] = args.lora_alpha
        
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config_overrides if config_overrides else None,
            lora_config=lora_overrides if lora_overrides else None,
            resume_from=args.resume_from,
            use_wandb=args.wandb,
            merge_and_save=args.merge,
        )


if __name__ == "__main__":
    main()