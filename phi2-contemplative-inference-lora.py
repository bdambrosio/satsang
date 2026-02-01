#!/usr/bin/env python3
"""
Phi-2 Contemplative LoRA Inference
==================================

Load a LoRA checkpoint and run inference with the fine-tuned model.
Can also run with base model only for comparison.

Usage:
    # With LoRA adapter (default)
    python phi2-contemplative-inference-lora.py \
        --checkpoint ./phi2-contemplative-lora \
        --base-model microsoft/phi-2 \
        --query "I feel lost in my practice."
    
    # Base model only (no LoRA)
    python phi2-contemplative-inference-lora.py \
        --base-model microsoft/phi-2 \
        --no-lora \
        --query "I feel lost in my practice."

Requirements:
    pip install torch transformers peft accelerate
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_lora_with_key_remap(base_model, checkpoint_dir: Path):
    """
    Load LoRA adapter with key remapping to handle torch.compile artifacts.
    
    The checkpoint may have keys like:
        _orig_mod.base_model.model.model.layers.0.mlp.fc1.lora_A.weight
    
    But PEFT expects:
        base_model.model.model.layers.0.mlp.fc1.lora_A.default.weight
    """
    from peft import PeftModel, LoraConfig, get_peft_model
    from safetensors.torch import load_file
    import json
    
    # Load adapter config
    config_path = checkpoint_dir / "adapter_config.json"
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
    model_keys = [k for k in model.state_dict().keys() if "lora" in k]
    logger.info(f"Model expects keys (sample): {model_keys[:2]}")
    
    # Load state dict with strict=False to allow partial loading
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    
    if missing:
        # Filter to only show lora-related missing keys
        lora_missing = [k for k in missing if "lora" in k]
        if lora_missing:
            logger.warning(f"Missing LoRA keys: {len(lora_missing)} (showing first 3: {lora_missing[:3]})")
    if unexpected:
        logger.info(f"Unexpected keys in checkpoint (ignored): {len(unexpected)}")
    
    return model


def load_model(
    base_model_name: str = "microsoft/phi-2",
    checkpoint_dir: Optional[Path] = None,
    use_lora: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: Optional[str] = None,
):
    """
    Load base model and optionally apply LoRA adapter.
    
    Args:
        base_model_name: Base model identifier (used if checkpoint doesn't specify base_model_name)
        checkpoint_dir: Path to LoRA checkpoint directory (required if use_lora=True)
        use_lora: Whether to load and apply LoRA adapter
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        device_map: Device placement strategy (e.g., "auto", "cuda:0")
    
    Returns:
        (model, tokenizer) tuple
    """
    # Determine device
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None
    
    # If using LoRA, check if checkpoint specifies base model
    actual_base_model = base_model_name
    if use_lora and checkpoint_dir is not None:
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                import json
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                # PEFT saves base_model_name_or_path in adapter config
                if "base_model_name_or_path" in adapter_config:
                    actual_base_model = adapter_config["base_model_name_or_path"]
                    logger.info(f"Using base_model from adapter config: {actual_base_model}")
            except Exception as e:
                logger.debug(f"Could not read base_model_name from adapter config: {e}, using {base_model_name}")
    
    logger.info(f"Loading base model: {actual_base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(actual_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Fix Phi-2 config compatibility (pad_token_id may not exist)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(actual_base_model, trust_remote_code=True)
    if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
        model_config.pad_token_id = model_config.eos_token_id
        logger.info(f"Set pad_token_id to eos_token_id ({model_config.eos_token_id})")
    
    # Load base model
    model_kwargs = {
        "config": model_config,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["load_in_8bit"] = True
        logger.info("Using 8-bit quantization")
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
        logger.info("Using 4-bit quantization")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        actual_base_model,
        **model_kwargs
    )
    
    logger.info(f"Base model loaded: {base_model.num_parameters():,} parameters")
    
    if use_lora:
        # Load LoRA adapter
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when use_lora=True")
        
        logger.info(f"Loading LoRA adapter from {checkpoint_dir}")
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Check if adapter_config.json exists (indicates LoRA checkpoint)
        adapter_config = checkpoint_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise ValueError(
                f"No adapter_config.json found in {checkpoint_dir}. "
                "Is this a LoRA checkpoint? Expected structure: "
                "checkpoint_dir/adapter_config.json, adapter_model.bin"
            )
        
        # Load with custom key remapping to handle torch.compile artifacts
        model = _load_lora_with_key_remap(base_model, checkpoint_dir)
        
        # Merge adapter into base model for inference (optional but faster)
        logger.info("Merging LoRA adapter into base model for inference...")
        model = model.merge_and_unload()
        
        logger.info("Model ready for inference (with LoRA adapter)")
    else:
        model = base_model
        logger.info("Model ready for inference (base model only, no LoRA)")
    
    return model, tokenizer


class Phi2InferenceProvider:
    """
    Simple inference provider for Phi-2 models with LoRA adapters.
    
    Usage:
        provider = Phi2InferenceProvider(
            base_model_name="microsoft/phi-2",
            checkpoint_dir=Path("./phi2-contemplative-lora/checkpoint-4000")
        )
        response = provider.generate("I feel lost.", temperature=0.7)
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/phi-2",
        checkpoint_dir: Optional[Path] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: Optional[str] = None,
    ):
        """Initialize the inference provider by loading model and tokenizer."""
        self.model, self.tokenizer = load_model(
            base_model_name=base_model_name,
            checkpoint_dir=checkpoint_dir,
            use_lora=(checkpoint_dir is not None),
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
    
    def generate(
        self,
        query: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response to a query.
        
        Args:
            query: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated response text
        """
        return generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            query=query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    
    def generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response from chat messages (ChatML format).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated response text
        """
        # Format messages using tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            query = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            # Fallback: simple formatting
            query = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return self.generate(
            query=query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )


def generate_response(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """
    Generate a response to a query.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        query: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated response text
    """
    # Tokenize input
    inputs = tokenizer(query, return_tensors="pt")
    
    # Move to same device as model
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "hf_device_map"):
        # Multi-GPU case
        device = next(iter(model.hf_device_map.values()))
    else:
        device = next(model.parameters()).device
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove input from response
    if response.startswith(query):
        response = response[len(query):].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Phi-2 Contemplative LoRA model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to LoRA checkpoint directory (required if --no-lora not set)"
    )
    
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Run with base model only, skip loading LoRA adapter"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/phi-2",
        help="Base model identifier (default: microsoft/phi-2). "
             "If checkpoint has adapter_config.json with base_model_name_or_path, that takes precedence."
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="I feel lost in my practice.",
        help="Query to generate response for"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9)"
    )
    
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device placement (default: auto)"
    )
    
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")
    
    use_lora = not args.no_lora
    if use_lora and args.checkpoint is None:
        raise ValueError("--checkpoint is required when using LoRA (omit --no-lora flag)")
    
    # Load model
    logger.info("=" * 60)
    if use_lora:
        logger.info("Loading model with LoRA adapter...")
    else:
        logger.info("Loading base model only (no LoRA)...")
    logger.info("=" * 60)
    
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        checkpoint_dir=args.checkpoint,
        use_lora=use_lora,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )
    
    # Run inference
    logger.info("=" * 60)
    logger.info("Running inference...")
    logger.info("=" * 60)
    logger.info(f"Query: {args.query}")
    logger.info("-" * 60)
    
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        query=args.query,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    
    logger.info(f"Response: {response}")
    logger.info("=" * 60)
    
    # Print formatted output
    print("\n" + "=" * 60)
    print("QUERY:")
    print("=" * 60)
    print(args.query)
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
