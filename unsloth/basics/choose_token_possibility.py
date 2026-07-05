# Run SmolLM2-1.7B-Instruct with Unsloth (inference only)
#
# Install:  pip install unsloth
# Run:      python choose_token_possibility
#
# --- If you get errors ---
# 1) "Unsloth cannot find any torch accelerator" → pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) "module 'torch' has no attribute 'int1'" → same as (1); need PyTorch 2.10+ with CUDA (cu128).
# 3) NumPy/soxr error: pip install "numpy<2"
# 4) "No module named 'tf_keras'": pip install tf-keras
# 5) ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2' (or similar wandb import fails):
#    → pip install --upgrade wandb   (unsloth/trl import wandb; a broken or old wandb causes this)

import torch
# PyTorch: used for GPU checks (torch.cuda), tensor device (.to("cuda")), and disabling gradients (torch.no_grad).

from unsloth import FastLanguageModel
# Unsloth's wrapper: loads models with optional 4-bit quantization and applies inference optimizations (faster generation).

# =============================================================================
# CONFIG — Edit these to change the prompt and how the model generates text
# =============================================================================

# PROMPT (str): The question or instruction you send to the model. The model will generate a reply to this text.
# Effect: More specific prompts usually get more focused answers; vague prompts may get generic or short replies.
PROMPT = "Who is Kapil Dev?"

# MODEL_NAME (str): Hugging Face model ID. "unsloth/SmolLM2-1.7B-Instruct" is the Unsloth-optimized 1.7B instruct model.
# Effect: Changing this loads a different model (e.g. another size or family); must be a model supported by Unsloth.
MODEL_NAME = "unsloth/SmolLM2-1.7B-Instruct"

# MAX_SEQ_LENGTH (int): Maximum total sequence length (prompt + generated tokens) the model can handle in one forward pass.
# Effect: Longer = can use longer prompts and longer replies, but uses more VRAM. 1024 is a safe default for 1.7B.
MAX_SEQ_LENGTH = 8000

# LOAD_IN_4BIT (bool): If True, load the model in 4-bit quantized form (QLoRA-style); if False, load in full precision.
# Effect: True = much less VRAM  but tiny quality loss; False = higher VRAM (~6–8 GB for 1.7B), slightly better quality.
LOAD_IN_4BIT = True

# MAX_NEW_TOKENS (int): Maximum number of tokens the model is allowed to generate after the prompt.
# Effect: Higher = longer possible answers but slower and more VRAM; lower = shorter, faster. 256 is enough for a paragraph.
MAX_NEW_TOKENS = 8000

# TEMPERATURE (float): Controls randomness of sampling. Only used when DO_SAMPLE is True.
# Effect: Lower (e.g. 0.3) = more deterministic, repetitive; higher (e.g. 1.0) = more varied, sometimes incoherent. 0.7 is a good balance.
TEMPERATURE = 1

# TOP_P (float): Nucleus sampling: only sample from the smallest set of tokens whose cumulative probability ≥ top_p. Used when DO_SAMPLE is True.
# Effect: Lower (e.g. 0.5) = more focused, conservative; higher (e.g. 1.0) = more diverse. 0.9 is common.
TOP_P = 0.33

# DO_SAMPLE (bool): If True, use random sampling (temperature, top_p); if False, always pick the most likely token (greedy decoding).
# Effect: True = varied answers each run; False = deterministic, same prompt → same answer (up to numerical precision).
DO_SAMPLE = True

# INSPECT_MODEL_AND_TOKENIZER (bool): If True, print detailed structure of model and tokenizer after loading (for learning).
# Set to False for normal fast runs; set to True to see vocab, config, parameter counts, and layer names.
INSPECT_MODEL_AND_TOKENIZER = True


def _inspect_tokenizer(tokenizer):
    """Print tokenizer structure: type, vocab size, special tokens, and a small encode/decode example."""
    print("\n" + "=" * 60)
    print("TOKENIZER INSPECTION")
    print("=" * 60)
    print(f"  Type: {type(tokenizer).__module__}.{type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    # Special token IDs (may be None)
    for name in ("pad_token", "eos_token", "bos_token", "unk_token"):
        attr = name.replace("_token", "_token_id")
        val = getattr(tokenizer, attr, None)
        token_str = getattr(tokenizer, name, None)
        print(f"  {name}: id={val!r}  repr={token_str!r}")
    # Sample vocab: first few and a few around the middle
    vocab = getattr(tokenizer, "get_vocab", None) or (lambda: {})
    if callable(vocab):
        vocab = vocab()
    if vocab:
        # get_vocab() returns token_string -> id; show id -> token (first 8 by id order)
        by_id = sorted((id_, token) for token, id_ in vocab.items())
        for tid, tstr in by_id[:8]:
            print(f"    {tid}: {repr(tstr)}")
    # Encode/decode example
    example = "Hello world"
    enc = tokenizer.encode(example, add_special_tokens=False)
    dec = tokenizer.decode(enc)
    print(f"  Encode('{example}'): {enc}")
    print(f"  Decode(enc): {repr(dec)}")
    print("=" * 60 + "\n")


def _inspect_model(model, tokenizer):
    """Print model structure: type, config, param count, and top-level layer names."""
    print("\n" + "=" * 60)
    print("MODEL INSPECTION")
    print("=" * 60)
    print(f"  Type: {type(model).__module__}.{type(model).__name__}")
    # Config (common keys for LLM)
    config = getattr(model, "config", None)
    if config is not None:
        print("  Config (main keys):")
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "max_position_embeddings"):
            if hasattr(config, key):
                print(f"    {key}: {getattr(config, key)}")
    # Total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total:,}  (trainable: {trainable:,})")
    print(f"  Approx. size @ 4 bytes: {total * 4 / 1e9:.2f} GB  |  @ 2 bytes (bf16): {total * 2 / 1e9:.2f} GB")
    # Top-level modules
    print("  Top-level module names:")
    for name, _ in list(model.named_children())[:20]:
        print(f"    - {name}")
    if len(list(model.named_children())) > 20:
        print(f"    ... and {len(list(model.named_children())) - 20} more")
    # If it has .model (e.g. PeftModel wraps model.model), show one level deeper
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        print("  Inner .model top-level (first 12):")
        for name, _ in list(inner.named_children())[:12]:
            print(f"    - {name}")
    # embed_tokens: token ID -> vector lookup table
    embed = getattr(model, "embed_tokens", None) or (getattr(inner, "embed_tokens", None) if inner is not None else None)
    if embed is not None:
        print("  embed_tokens:")
        print(f"    type: {type(embed).__module__}.{type(embed).__name__}")
        if hasattr(embed, "weight"):
            w = embed.weight
            print(f"    weight.shape: {tuple(w.shape)}  (vocab_size, hidden_size)")
            print(f"    weight.dtype: {w.dtype}")
            # Vector for "Hello world": encode then embed
            ids = tokenizer.encode("Hello world", add_special_tokens=False)
            if ids:
                id_tensor = torch.tensor([ids], device=w.device)
                with torch.no_grad():
                    vecs = embed(id_tensor)
                print('  Embedding vectors for "Hello world":')
                print(f"    token IDs: {ids}")
                print(f"    output shape: {tuple(vecs.shape)}  (batch=1, num_tokens, hidden_size)")
                for i, tid in enumerate(ids):
                    v = vecs[0, i]
                    head = v[:8].tolist()
                    rest = v.numel() - 8
                    print(f"    token {i} (id={tid}): first 8 dims {[round(x, 5) for x in head]}{f' ... +{rest} more' if rest > 0 else ''}")
                # Position: show position index per token and position vectors if available
                num_tokens = len(ids)
                print('  Position for "Hello world" tokens:')
                for i in range(num_tokens):
                    print(f"    token {i} (id={ids[i]}): position index = {i}")
                # Try to get additive position embeddings (LLaMA/SmolLM use RoPE; some models have embed_positions)
                base = inner if inner is not None else model
                pos_embed = getattr(base, "embed_positions", None) or getattr(base, "position_embeddings", None)
                if pos_embed is not None and hasattr(pos_embed, "weight"):
                    pos_w = pos_embed.weight
                    print(f"    Position embedding layer shape: {tuple(pos_w.shape)}")
                    for i in range(min(num_tokens, pos_w.shape[0])):
                        pv = pos_w[i]
                        head = pv[:8].tolist()
                        rest = pv.numel() - 8
                        print(f"    position {i} vector (first 8 dims): {[round(x, 5) for x in head]}{f' ... +{rest} more' if rest > 0 else ''}")
                else:
                    # RoPE: position is applied in attention; optionally show inv_freq from first layer
                    print("    (This model uses RoPE; position is encoded in attention, not as additive vectors.)")
                    layers = getattr(base, "layers", None)
                    if layers is None:
                        layers = getattr(getattr(base, "model", None), "layers", None)
                    # RoPE can be on the model (base.rotary_emb) or on each layer's self_attn
                    rope = getattr(base, "rotary_emb", None)
                    if rope is None and layers is not None and len(layers) > 0:
                        first_layer = layers[0]
                        rope = getattr(getattr(first_layer, "self_attn", None), "rotary_emb", None)
                    inv = None
                    if rope is not None and hasattr(rope, "inv_freq"):
                        inv = rope.inv_freq
                    if inv is None and config is not None and layers is not None and len(layers) > 0:
                        # Fallback: compute inv_freq from config (standard Llama RoPE)
                        head_dim = getattr(config, "head_dim", None) or (getattr(config, "hidden_size", 2048) // getattr(config, "num_attention_heads", 32))
                        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=w.device).float() / head_dim))
                    if inv is not None:
                        n = min(4, inv.numel())
                        print(f"    RoPE inv_freq (first {n}): {[round(x, 6) for x in inv[:n].tolist()]}")
                        # How position gets into the next layer: show input to first layer and RoPE effect
                        print('  Input to first transformer layer:')
                        print('    The layer receives the embedding vectors above (no position added yet).')
                        print('    RoPE is applied inside the layer to Q and K. Below: same embedding rotated by position.')
                        # Apply RoPE to first 4 dims (2 pairs) of token 0 and token 1 for positions 0 and 1
                        with torch.no_grad():
                            inv_freq = inv.float() if inv.dtype != torch.float32 else inv
                            for pos in range(num_tokens):
                                # cos/sin for this position: shape (num_pairs,)
                                t = torch.tensor([pos], device=inv_freq.device, dtype=inv_freq.dtype) * inv_freq
                                cos_t = t.cos()
                                sin_t = t.sin()
                                v = vecs[0, pos]
                                # First 4 dims = 2 pairs; rotate each pair
                                rotated = []
                                for pair_idx in range(min(2, len(cos_t))):
                                    d0, d1 = v[2 * pair_idx].item(), v[2 * pair_idx + 1].item()
                                    c, s = cos_t[pair_idx].item(), sin_t[pair_idx].item()
                                    r0 = d0 * c - d1 * s
                                    r1 = d0 * s + d1 * c
                                    rotated.extend([round(r0, 5), round(r1, 5)])
                                print(f'    token {pos} (position {pos}) embedding first 4 dims: {[round(x, 5) for x in v[:4].tolist()]}')
                                print(f'    token {pos} after RoPE (first 4 dims): {rotated}')
    print("=" * 60 + "\n")


def main():
    # --- GPU check ---
    # torch.cuda.is_available() is True only if PyTorch was built with CUDA and sees an NVIDIA GPU. Unsloth needs a GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA GPU. PyTorch did not detect one.")

    # --- Load model and tokenizer ---
    # FastLanguageModel.from_pretrained: downloads (or uses cache) and loads the model and its tokenizer.
    # - model_name: Hugging Face repo id (e.g. unsloth/SmolLM2-1.7B-Instruct).
    # - max_seq_length: Passed to the model config; caps context length and affects some internal buffers.
    # - dtype=None: Let Unsloth choose the numeric format (usually bfloat16). dtype = how each weight is stored/used: e.g. float32 (4 bytes), bfloat16 (2 bytes, FP32-like range). With load_in_4bit=True, 4-bit is storage; dtype is used for the actual computation after dequantization.
    # - load_in_4bit=True: Store most weights in 4-bit (half byte per weight) to save VRAM. Weights are dequantized on the fly for math; that computation uses the dtype above (e.g. bfloat16). False = full precision, dtype applies to all weights.
    # Returns: (model, tokenizer). Model is on CPU or GPU depending on Unsloth; we move inputs to "cuda" later.
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto → usually bfloat16 (16-bit float, FP32-like range, 2 bytes per weight)
        load_in_4bit=LOAD_IN_4BIT,
    )

    if INSPECT_MODEL_AND_TOKENIZER:
        _inspect_tokenizer(tokenizer)
        _inspect_model(model, tokenizer)

    # FastLanguageModel.for_inference(model): Switches the model to inference mode and applies Unsloth's speed optimizations
    # (e.g. kernel fusions, different attention path). Call this before generate(); do not use for training.
    FastLanguageModel.for_inference(model)

    # --- Build the chat prompt ---
    # SmolLM2-Instruct expects messages in chat format. We send a single "user" message; the model will reply as "assistant".
    # messages: list of dicts with "role" ("user" / "assistant" / "system") and "content" (str). One user message = one turn.
    messages = [{"role": "user", "content": PROMPT}]

    # tokenizer.apply_chat_template: Converts messages into the exact string format the model was trained on (e.g. <|im_start|>user\n...<|im_end|>).
    # - tokenize=False: We want the prompt as a string so we can tokenize it together with padding; if True, returns token ids.
    # - add_generation_prompt=True: Appends the assistant turn start (e.g. <|im_start|>assistant\n) so the model knows to generate a reply.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    # tokenizer(prompt, ...): Converts the prompt string into token IDs and an attention mask.
    # return_tensors="pt": Return PyTorch tensors (not lists or NumPy). .to("cuda") moves them to the GPU so the model runs on GPU.
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print("Inputs:", inputs)

    # --- Inspect lm_head logits (one forward pass, no generation) ---
    if INSPECT_MODEL_AND_TOKENIZER:
        with torch.no_grad():
            model_output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = model_output.logits  # shape: (batch=1, seq_len, vocab_size)

        print("\n" + "=" * 60)
        print("LM_HEAD LOGITS INSPECTION")
        print("=" * 60)
        print(f"  logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")

        last_logits = logits[0, -1, :]  # shape: (vocab_size,) — prediction for the NEXT token
        print(f"  Last position logits shape: {tuple(last_logits.shape)}  (one score per vocab word)")
        print(f"  Min score: {last_logits.min().item():.4f}   Max score: {last_logits.max().item():.4f}")

        probs = torch.nn.functional.softmax(last_logits.float(), dim=0)

        top_k = 20
        top_scores, top_ids = torch.topk(last_logits, top_k)
        print(f"\n  Top {top_k} predicted next tokens (from {last_logits.shape[0]} candidates):")
        print(f"  {'Rank':<6} {'ID':<8} {'Token':<20} {'Logit':>10} {'Probability':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*20} {'-'*10} {'-'*12}")
        for rank, (score, tid) in enumerate(zip(top_scores, top_ids), 1):
            token_str = tokenizer.decode([tid.item()])
            prob = probs[tid.item()].item()
            print(f"  {rank:<6} {tid.item():<8} {repr(token_str):<20} {score.item():>10.4f} {prob:>11.4%}")

        greedy_id = last_logits.argmax().item()
        greedy_token = tokenizer.decode([greedy_id])
        print(f"\n  Greedy pick (argmax): ID={greedy_id}  token={repr(greedy_token)}  score={last_logits[greedy_id].item():.4f}")
        print("=" * 60 + "\n")

    # torch.no_grad(): Disables gradient computation. We are only doing inference, so gradients are not needed; this saves memory and speed.
    with torch.no_grad():
        # model.generate: Autoregressively generates token IDs until max_new_tokens or EOS.
        # - **inputs: Unpacks input_ids and attention_mask (and optionally other keys) as keyword arguments.
        # - max_new_tokens: Upper limit on how many new tokens to generate (excluding the prompt length).
        # - temperature: Scaling of logits before softmax; None when do_sample=False.
        # - top_p: Nucleus sampling threshold; None when do_sample=False.
        # - do_sample: True = sample from the distribution; False = take argmax each step (greedy).
        # - pad_token_id: Token ID to use when padding sequences. Must match the tokenizer; use eos_token_id if pad is not set.
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # --- Decode the model's reply (only the new tokens) ---
    # inputs["input_ids"].shape[1]: Number of tokens in the prompt (batch size is shape[0]). We want to skip these when decoding.
    prompt_length = inputs["input_ids"].shape[1]

    # outputs[0]: First (and only) sequence in the batch. outputs[0][prompt_length:]: only the newly generated token IDs.
    response_ids = outputs[0][prompt_length:]
    print("Response IDs:", response_ids)
    # tokenizer.decode: Converts token IDs back to a string. skip_special_tokens=True removes tokens like <|im_end|>, <|endoftext|>.
    # .strip(): Removes leading/trailing whitespace and newlines.
    answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Print the prompt and the model's answer so you can see the result.
    print("\n--- Prompt ---")
    print(PROMPT)
    print("\n--- Model answer ---")
    print(answer)
    print()


if __name__ == "__main__":
    # Run main() only when this file is executed directly (e.g. python smollm2_unsloth_inference.py), not when imported as a module.
    main()
