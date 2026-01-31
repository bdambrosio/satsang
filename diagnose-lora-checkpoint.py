#!/usr/bin/env python3
"""
Diagnose a PEFT LoRA checkpoint.

This script is meant to answer:
- What LoRA tensors are actually present in adapter_model.safetensors?
- Which layers/modules are missing vs what adapter_config.json implies?
- Are key naming issues present (_orig_mod, missing .default, etc.)?

Usage:
  python diagnose-lora-checkpoint.py \
    --checkpoint-dir ./phi2-contemplative-lora/checkpoint-1000

  # JSON output (machine-readable)
  python diagnose-lora-checkpoint.py \
    --checkpoint-dir ./phi2-contemplative-lora/checkpoint-1000 \
    --json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


LORA_A_RE = re.compile(r"\.lora_A(\.default)?\.weight$")
LORA_B_RE = re.compile(r"\.lora_B(\.default)?\.weight$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_get_num_layers(base_model_name_or_path: Optional[str]) -> Optional[int]:
    if not base_model_name_or_path:
        return None
    try:
        from transformers import AutoConfig
    except Exception:
        return None

    try:
        cfg = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if isinstance(v, int) and v > 0:
                    return v
        return None
    except Exception:
        return None


def _load_safetensors_keys(adapter_safetensors: Path) -> list[str]:
    from safetensors.torch import load_file

    state = load_file(str(adapter_safetensors))
    return list(state.keys())


def _analyze_keys(
    keys: list[str],
    target_modules: list[str],
    expected_num_layers: Optional[int],
) -> dict[str, Any]:
    # Basic key hygiene signals
    has_orig_mod = any("_orig_mod." in k for k in keys)
    has_default_suffix = any(".lora_A.default.weight" in k or ".lora_B.default.weight" in k for k in keys)
    has_non_default_suffix = any(k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight") for k in keys)

    # Collect observed structure
    layers_seen: set[int] = set()
    modules_seen: set[str] = set()

    # layer -> module -> {A:count, B:count}
    coverage: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"A": 0, "B": 0}))

    # Also count by module regardless of layer
    module_totals = defaultdict(lambda: {"A": 0, "B": 0})

    # Track keys that look like LoRA but don’t match expected module strings
    lora_like_keys = []

    for k in keys:
        is_a = bool(LORA_A_RE.search(k)) or k.endswith(".lora_A.weight")
        is_b = bool(LORA_B_RE.search(k)) or k.endswith(".lora_B.weight")
        if not (is_a or is_b):
            continue

        lora_like_keys.append(k)

        m = LAYER_RE.search(k)
        layer = int(m.group(1)) if m else None
        if layer is not None:
            layers_seen.add(layer)

        # Determine which target module this key belongs to (best-effort)
        matched_module = None
        for tm in target_modules:
            # Require a dot boundary to reduce accidental matches
            if f".{tm}." in k:
                matched_module = tm
                break
        if matched_module is None:
            # fallback: sometimes module is at end like "...q_proj.lora_A..."
            for tm in target_modules:
                if k.endswith(f".{tm}.lora_A.weight") or k.endswith(f".{tm}.lora_B.weight"):
                    matched_module = tm
                    break

        if matched_module is None:
            # Keep going, but record as unassigned
            continue

        modules_seen.add(matched_module)
        if is_a:
            module_totals[matched_module]["A"] += 1
            if layer is not None:
                coverage[layer][matched_module]["A"] += 1
        if is_b:
            module_totals[matched_module]["B"] += 1
            if layer is not None:
                coverage[layer][matched_module]["B"] += 1

    # Compute expected counts (best effort)
    expected = None
    missing_summary = None

    if expected_num_layers is not None and target_modules:
        expected = {
            "layers": expected_num_layers,
            "target_modules": target_modules,
            "expected_lora_tensors": expected_num_layers * len(target_modules) * 2,  # A+B per layer per module
        }

        missing_by_layer: dict[int, dict[str, list[str]]] = {}
        for layer in range(expected_num_layers):
            layer_missing: dict[str, list[str]] = {}
            for tm in target_modules:
                a_present = coverage[layer][tm]["A"] > 0
                b_present = coverage[layer][tm]["B"] > 0
                missing = []
                if not a_present:
                    missing.append("A")
                if not b_present:
                    missing.append("B")
                if missing:
                    layer_missing[tm] = missing
            if layer_missing:
                missing_by_layer[layer] = layer_missing

        # Useful aggregates
        fully_covered_layers = 0
        partially_covered_layers = 0
        empty_layers = 0
        for layer in range(expected_num_layers):
            layer_cov = coverage.get(layer, {})
            present_pairs = 0
            for tm in target_modules:
                if layer_cov.get(tm, {}).get("A", 0) > 0 and layer_cov.get(tm, {}).get("B", 0) > 0:
                    present_pairs += 1
            if present_pairs == 0:
                empty_layers += 1
            elif present_pairs == len(target_modules):
                fully_covered_layers += 1
            else:
                partially_covered_layers += 1

        missing_summary = {
            "fully_covered_layers": fully_covered_layers,
            "partially_covered_layers": partially_covered_layers,
            "empty_layers": empty_layers,
            "layers_with_any_lora": sorted(layers_seen),
            "missing_by_layer": missing_by_layer,  # can be large
        }

    # Summarize layer coverage compactly: for each layer count modules with both A+B
    compact_layer_stats = []
    if expected_num_layers is not None and target_modules:
        for layer in range(expected_num_layers):
            present_pairs = 0
            for tm in target_modules:
                if coverage[layer][tm]["A"] > 0 and coverage[layer][tm]["B"] > 0:
                    present_pairs += 1
            compact_layer_stats.append({"layer": layer, "modules_with_A_and_B": present_pairs})

    return {
        "key_counts": {
            "total_keys": len(keys),
            "lora_like_keys": len(lora_like_keys),
        },
        "naming_signals": {
            "has__orig_mod_prefix": has_orig_mod,
            "has_.default_in_lora_keys": has_default_suffix,
            "has_non_default_lora_suffix": has_non_default_suffix,
        },
        "observed": {
            "layers_seen": sorted(layers_seen),
            "modules_seen": sorted(modules_seen),
            "module_totals": module_totals,
        },
        "expected": expected,
        "missing_summary": missing_summary,
        "compact_layer_stats": compact_layer_stats,
        "examples": {
            "first_10_keys": keys[:10],
            "first_10_lora_like_keys": lora_like_keys[:10],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose a PEFT LoRA checkpoint directory")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to checkpoint dir (contains adapter_config.json)")
    p.add_argument(
        "--adapter-file",
        type=Path,
        default=None,
        help="Path to adapter safetensors (default: <checkpoint-dir>/adapter_model.safetensors)",
    )
    p.add_argument(
        "--compare-bak",
        action="store_true",
        help="Also analyze the original adapter .bak file side-by-side",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model name/path (default: read from adapter_config.json base_model_name_or_path)",
    )
    p.add_argument("--json", action="store_true", help="Output JSON report")
    p.add_argument("--max-missing-layers", type=int, default=3, help="When not --json, print missing details for first N layers")
    args = p.parse_args()

    ckpt = args.checkpoint_dir
    adapter_cfg_path = ckpt / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise SystemExit(f"Missing {adapter_cfg_path}")

    adapter_cfg = _read_json(adapter_cfg_path)
    target_modules = adapter_cfg.get("target_modules") or []
    if not isinstance(target_modules, list):
        raise SystemExit(f"adapter_config.json target_modules is not a list: {type(target_modules)}")

    base_model_name = args.base_model or adapter_cfg.get("base_model_name_or_path")
    expected_layers = _maybe_get_num_layers(base_model_name)

    adapter_file = args.adapter_file or (ckpt / "adapter_model.safetensors")
    if not adapter_file.exists():
        raise SystemExit(f"Missing adapter file: {adapter_file}")

    keys = _load_safetensors_keys(adapter_file)
    bak_file: Optional[Path] = None
    bak_keys: Optional[list[str]] = None
    if args.compare_bak:
        # Prefer "<adapter-file>.bak" if present, else "adapter_model.safetensors.bak"
        candidate_1 = Path(str(adapter_file) + ".bak")
        candidate_2 = ckpt / "adapter_model.safetensors.bak"
        if candidate_1.exists():
            bak_file = candidate_1
        elif candidate_2.exists():
            bak_file = candidate_2
        else:
            raise SystemExit(
                "Requested --compare-bak but no .bak found. "
                f"Tried {candidate_1} and {candidate_2}"
            )
        bak_keys = _load_safetensors_keys(bak_file)

    report = {
        "checkpoint_dir": str(ckpt),
        "adapter_file": str(adapter_file),
        "adapter_file_bak": str(bak_file) if bak_file is not None else None,
        "base_model_name_or_path": base_model_name,
        "adapter_config": {
            "peft_version": adapter_cfg.get("peft_version"),
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "lora_dropout": adapter_cfg.get("lora_dropout"),
            "target_modules": target_modules,
        },
        "expected_num_layers": expected_layers,
        "analysis": _analyze_keys(keys, target_modules, expected_layers),
        "analysis_bak": _analyze_keys(bak_keys, target_modules, expected_layers) if bak_keys is not None else None,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=list))
        return

    # Human-readable summary
    a = report["analysis"]
    print("=" * 80)
    print("LoRA checkpoint diagnostic")
    print("=" * 80)
    print(f"Checkpoint dir: {report['checkpoint_dir']}")
    print(f"Adapter file:   {report['adapter_file']}")
    print(f"Base model:     {report['base_model_name_or_path']}")
    print(f"Expected layers (best effort): {report['expected_num_layers']}")
    print()
    print("Adapter config:")
    print(f"  peft_version: {report['adapter_config'].get('peft_version')}")
    print(f"  r:            {report['adapter_config'].get('r')}")
    print(f"  lora_alpha:   {report['adapter_config'].get('lora_alpha')}")
    print(f"  targets:      {report['adapter_config'].get('target_modules')}")
    print()
    print("Key counts:")
    print(f"  total keys:       {a['key_counts']['total_keys']}")
    print(f"  lora-like keys:   {a['key_counts']['lora_like_keys']}")
    print()
    print("Naming signals:")
    print(f"  has _orig_mod.:            {a['naming_signals']['has__orig_mod_prefix']}")
    print(f"  has .default in lora keys: {a['naming_signals']['has_.default_in_lora_keys']}")
    print(f"  has non-default suffixes:  {a['naming_signals']['has_non_default_lora_suffix']}")
    print()
    print("Observed coverage:")
    print(f"  layers with any LoRA keys: {a['observed']['layers_seen'][:20]}{' ...' if len(a['observed']['layers_seen']) > 20 else ''}")
    print(f"  modules seen:              {a['observed']['modules_seen']}")
    print(f"  per-module totals (A/B):   {dict(a['observed']['module_totals'])}")
    print()

    if report.get("analysis_bak") is not None:
        b = report["analysis_bak"]
        print("-" * 80)
        print("Comparison vs .bak (original pre-inference_patch):")
        print(f"  bak file: {report.get('adapter_file_bak')}")
        print()
        print("  Key counts (current vs bak):")
        print(f"    total keys:     {a['key_counts']['total_keys']} vs {b['key_counts']['total_keys']}")
        print(f"    lora-like keys: {a['key_counts']['lora_like_keys']} vs {b['key_counts']['lora_like_keys']}")
        print()
        print("  Naming signals (current vs bak):")
        print(f"    has _orig_mod.:            {a['naming_signals']['has__orig_mod_prefix']} vs {b['naming_signals']['has__orig_mod_prefix']}")
        print(f"    has .default in lora keys: {a['naming_signals']['has_.default_in_lora_keys']} vs {b['naming_signals']['has_.default_in_lora_keys']}")
        print(f"    has non-default suffixes:  {a['naming_signals']['has_non_default_lora_suffix']} vs {b['naming_signals']['has_non_default_lora_suffix']}")
        print()
        print("  Coverage (current vs bak):")
        print(f"    layers with any LoRA keys: {len(a['observed']['layers_seen'])} vs {len(b['observed']['layers_seen'])}")
        print(f"    modules seen:              {a['observed']['modules_seen']} vs {b['observed']['modules_seen']}")
        print()

    if a["missing_summary"] is None:
        print("Missing-summary: cannot compute (unknown num layers or missing target_modules)")
        return

    ms = a["missing_summary"]
    print("Missing-summary (expected A+B per module per layer):")
    print(f"  fully covered layers:     {ms['fully_covered_layers']}")
    print(f"  partially covered layers: {ms['partially_covered_layers']}")
    print(f"  empty layers:             {ms['empty_layers']}")
    print()

    # Print details for first N layers that have missing
    missing_by_layer = ms["missing_by_layer"]
    if missing_by_layer:
        print(f"Example missing details (first {args.max_missing_layers} layers with missing):")
        shown = 0
        for layer in sorted(missing_by_layer.keys()):
            print(f"  layer {layer}: {missing_by_layer[layer]}")
            shown += 1
            if shown >= args.max_missing_layers:
                break
    else:
        print("No missing modules detected (unexpected given earlier warnings).")


if __name__ == "__main__":
    main()

