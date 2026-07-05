# =============================================================================
# Chat with your fine-tuned SmolLM2 model
# =============================================================================
#
# This script loads the fine-tuned model (base + LoRA adapters from
# 38_finetune.py) and lets you ask questions interactively.
#
# Run:
#   python using_trained_model.py
# =============================================================================

import os
import torch
from unsloth import FastLanguageModel

# --- Config ------------------------------------------------------------------

ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_model")
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 8000

# --- Load model --------------------------------------------------------------

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            f"Run 38_finetune.py first."
        )

    print("Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    print("Model ready.\n")
    return model, tokenizer


def generate(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    response_ids = outputs[0][prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


# --- Main --------------------------------------------------------------------

def main():
    model, tokenizer = load_model()

    print("=" * 50)
    print("  Fine-tuned SmolLM2 — Interactive Chat")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        answer = generate(model, tokenizer, question)
        print(f"\nAssistant: {answer}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
