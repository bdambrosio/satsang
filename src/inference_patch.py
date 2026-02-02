from safetensors.torch import load_file, save_file

checkpoint = "phi2-contemplative-lora/checkpoint-1000/adapter_model.safetensors"
backup = checkpoint + ".bak"

# Load from backup if it exists (original), else from checkpoint
import os
source = backup if os.path.exists(backup) else checkpoint
state_dict = load_file(source)

# Rename keys
# Original: _orig_mod.base_model.model.model.layers.0.mlp.fc1.lora_A.weight
# Target:   model.layers.0.mlp.fc1.lora_A.default.weight
new_state = {}
for key, value in state_dict.items():
    new_key = key
    # Remove _orig_mod. and base_model. prefixes
    # Original: _orig_mod.base_model.model.model.layers...
    # Target:   model.model.layers... (PEFT strips base_model. when saving)
    new_key = new_key.replace("_orig_mod.", "")
    if new_key.startswith("base_model."):
        new_key = new_key[len("base_model."):]
    # Add .default before .weight for lora keys
    new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
    new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
    new_state[new_key] = value

# Save backup (only if not already backed up) and overwrite
import shutil
if not os.path.exists(backup):
    shutil.copy(checkpoint, backup)
save_file(new_state, checkpoint)

print("Fixed!")
print(f"Sample key: {list(new_state.keys())[0]}")