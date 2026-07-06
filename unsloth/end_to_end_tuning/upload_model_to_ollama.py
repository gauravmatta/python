# =============================================================================
# Export fine-tuned SmolLM2 to Ollama
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads the base SmolLM2-1.7B-Instruct model + your LoRA adapters.
#   2. MERGES the LoRA adapters into the base weights, producing a single
#      complete model (no more separate adapter files).
#   3. Saves the merged model in Hugging Face safetensors format.
#   4. Creates an Ollama "Modelfile" that tells Ollama how to import it.
#
# AFTER THIS SCRIPT:
#   Run one command to import into Ollama (printed at the end).
#   Ollama handles the GGUF conversion and q4_K_M quantization internally.
#
# WHY NOT USE save_pretrained_gguf()?
#   Unsloth's built-in GGUF export tries to build llama.cpp from source,
#   which requires Linux tools (apt-get, cmake, gcc). On Windows, it fails.
#   Instead, we save as safetensors and let Ollama do the conversion — this
#   works on any OS without extra build tools.
#
# Run:
#   python upload_model_to_ollama.py
# =============================================================================

import os
import torch
from unsloth import FastLanguageModel

# --- Config ------------------------------------------------------------------

# ADAPTER_DIR: Where 38_finetune.py saved the LoRA adapters.
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_model")

# MERGED_DIR: Where to save the fully merged model (base + LoRA combined).
# This will be a ~3.4 GB folder containing safetensors files + tokenizer.
MERGED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_model_merged")

# MODELFILE_PATH: The Ollama Modelfile that we auto-generate.
MODELFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelfile")

# MAX_SEQ_LENGTH: Must match the value used during training.
MAX_SEQ_LENGTH = 2048

# LOAD_IN_4BIT: Must match training setting. False = full 16-bit precision,
# which produces an exact merge (no dequantization error).
LOAD_IN_4BIT = False

# OLLAMA_MODEL_NAME: The name your model will have in Ollama.
# After import, you'll use: ollama run namaste-jaishankar
OLLAMA_MODEL_NAME = "namaste-jaishankar"

# QUANTIZATION: Ollama quantization level applied during import.
# q4_K_M = 4-bit k-quant medium (~1 GB, great quality-to-size ratio).
OLLAMA_QUANTIZATION = "q4_K_M"


# --- Main --------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            f"Run 38_finetune.py first."
        )

    # =========================================================================
    # STEP 1: Load the fine-tuned model (base + LoRA adapters)
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading fine-tuned model (base + LoRA adapters)")
    print("=" * 60)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("  Model loaded successfully.")

    # =========================================================================
    # STEP 2: Merge LoRA adapters into the base model and save
    # =========================================================================
    # save_pretrained_merged() does:
    #   1. Takes each layer that has a LoRA adapter (A and B matrices).
    #   2. Computes: merged_weight = original_weight + (A × B × alpha/r)
    #   3. Replaces the original weight with the merged result.
    #   4. Removes the LoRA adapter (no longer needed — it's baked in).
    #   5. Saves the full model as safetensors files (16-bit precision).
    #
    # The result is a complete, standalone model that doesn't need the
    # adapter files anymore. It contains all 1.7B parameters with your
    # personal knowledge permanently embedded in the weights.
    print("\n" + "=" * 60)
    print("STEP 2: Merging LoRA into base model + saving (16-bit)")
    print("=" * 60)
    print(f"  Output: {MERGED_DIR}")
    print("  This may take 1-2 minutes...\n")

    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print("  Merge complete.")

    # =========================================================================
    # STEP 3: Create the Ollama Modelfile
    # =========================================================================
    # A Modelfile tells Ollama:
    #   FROM  → where to find the model weights
    #   TEMPLATE → how to format chat messages (must match training format)
    #   PARAMETER → default generation settings
    #
    # SmolLM2-Instruct uses the ChatML template:
    #   <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n...
    print("\n" + "=" * 60)
    print("STEP 3: Creating Ollama Modelfile")
    print("=" * 60)

    modelfile_content = f"""FROM {MERGED_DIR}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"  Modelfile saved to: {MODELFILE_PATH}")

    # =========================================================================
    # STEP 4: Print the Ollama import command
    # =========================================================================
    print("\n" + "=" * 60)
    print("DONE! Now import into Ollama:")
    print("=" * 60)
    print()
    print(f"  ollama create {OLLAMA_MODEL_NAME} --quantize {OLLAMA_QUANTIZATION} -f \"{MODELFILE_PATH}\"")
    print()
    print("  This command tells Ollama to:")
    print(f"    1. Read the merged model from {MERGED_DIR}")
    print(f"    2. Convert to GGUF format with {OLLAMA_QUANTIZATION} quantization (~1 GB)")
    print(f"    3. Register it as '{OLLAMA_MODEL_NAME}'")
    print()
    print("  After that, use your model:")
    print(f"    ollama run {OLLAMA_MODEL_NAME}")
    print()
    print("  Other useful commands:")
    print("    ollama list                    — see all your models")
    print(f"    ollama rm {OLLAMA_MODEL_NAME}  — delete the model")
    print()

    # =========================================================================
    # STEP 5: How to upload your model to the Ollama website
    # =========================================================================
    # By default, your model only exists on your local machine. To share it
    # publicly so anyone in the world can use it, you can push it to the
    # Ollama registry (ollama.com). This is similar to pushing code to GitHub.
    print("=" * 60)
    print("OPTIONAL: Upload your model to ollama.com")
    print("=" * 60)
    print()
    print("  Your model is currently LOCAL only. To share it publicly:")
    print()
    print("  1. Create an account at https://ollama.com/signup")
    print()
    print("  2. Sign in from the terminal:")
    print("       ollama login")
    print()
    print("  3. Tag your model with your Ollama username:")
    print(f"       ollama cp {OLLAMA_MODEL_NAME} YOUR_USERNAME/{OLLAMA_MODEL_NAME}")
    print()
    print("  4. Push to the Ollama registry:")
    print(f"       ollama push YOUR_USERNAME/{OLLAMA_MODEL_NAME}")
    print()
    print("  After pushing, anyone can use your model with:")
    print(f"       ollama run YOUR_USERNAME/{OLLAMA_MODEL_NAME}")
    print()
    print("  Your model page will be at:")
    print(f"       https://ollama.com/YOUR_USERNAME/{OLLAMA_MODEL_NAME}")
    print()
    print("  NOTE: The uploaded model will be PUBLIC (anyone can download it).")
    print("        Make sure your training data doesn't contain sensitive info.")
    print("=" * 60)

# ollama cp namaste-jaishankar namaste-jaishankar/namaste-jaishankar
# ollama push namaste-jaishankar/namaste-jaishankar
if __name__ == "__main__":
    main()
