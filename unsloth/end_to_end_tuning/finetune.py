# =============================================================================
# Fine-tune SmolLM2-1.7B-Instruct with personal data using Unsloth + QLoRA
# =============================================================================
#
# WHAT IS FINE-TUNING?
#   A pre-trained LLM (like SmolLM2) already knows language, grammar, and
#   general knowledge. But it doesn't know about YOU. Fine-tuning teaches the
#   model new facts by showing it Q&A examples (your personal data) and
#   adjusting a small portion of its weights so it can answer correctly.
#
# WHAT IS QLoRA?
#   Instead of updating ALL 1.7 billion parameters (which would need ~14 GB
#   of VRAM), QLoRA keeps the base model frozen in 4-bit precision and only
#   trains tiny "adapter" matrices attached to each layer. This cuts VRAM
#   to ~4-6 GB while achieving nearly the same quality as full fine-tuning.
#
# DATASET STRATEGY:
#   - TRAIN on 100% of personal_data.json (no knowledge lost).
#   - VALIDATE on a separate file (personal_data_val.json) containing
#     general-knowledge questions the base model already answers correctly.
#     This acts as a "catastrophic forgetting detector": if the model starts
#     losing its general abilities, the validation loss will rise.
#
# Install (if not already):
#   pip install unsloth
#   pip install trl datasets
#
# Run:
#   python finetune.py
#
# After training, the LoRA adapters are saved to ./personal_model/
# Use test_finetuned.py to test the fine-tuned model.
# =============================================================================


# --- IMPORTS ----------------------------------------------------------------
# json: for reading the .json dataset files.
# os: for building file paths that work on any operating system.
# torch: PyTorch — the deep learning framework that runs everything on the GPU.
# FastLanguageModel: Unsloth's optimized wrapper to load and patch LLMs for
#   faster training (2x speed boost, lower VRAM usage).
# SFTTrainer: "Supervised Fine-Tuning Trainer" from the trl library.
#   It handles the training loop: feeding data, computing loss, updating weights.
# SFTConfig: configuration object that holds all training hyperparameters.
# TrainerCallback: base class for creating custom hooks that run during training
#   (we use it to implement early stopping).
# Dataset: a Hugging Face object that stores our training examples in a format
#   the trainer can iterate over efficiently.
import json
import os
import sys

# Force CUDA detection before Unsloth imports
# On Windows, PyTorch may not detect CUDA in fresh sessions.
# This workaround forces CUDA initialization.
import torch
_ = torch.cuda.is_available()  # Trigger CUDA initialization
if not torch.cuda.is_available():
    print("WARNING: CUDA not detected. Attempting to reinitialize...")
    # Try forcing CUDA contexts
    try:
        torch.cuda.init()
    except:
        pass

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from datasets import Dataset, disable_caching

# Disable caching/fingerprinting globally
disable_caching()


# =============================================================================
# CONFIG — All tunable settings in one place
# =============================================================================

# DATASET_PATH: Full path to your training data file.
# This JSON file contains conversations like:
#   [{"messages": [{"role":"user","content":"Who are you?"}, {"role":"assistant","content":"I am..."}]}]
# The model will learn to reproduce the assistant's answers for each question.
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_data.json")

# VAL_DATASET_PATH: Full path to the validation data file.
# Contains general-knowledge Q&A (math, geography, science) that the base model
# already answers correctly. The model never trains on these — they only measure
# whether training is damaging the model's existing knowledge ("catastrophic forgetting").
VAL_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_data_val.json")

# OUTPUT_DIR: Where to save the trained LoRA adapter files after fine-tuning.
# These files are small (~250 MB) and contain ONLY the new learned weights,
# not the entire 1.7B model. To use them, you load the base model + these adapters.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_model")

# MODEL_NAME: Which pre-trained model to start from.
# "unsloth/SmolLM2-1.7B-Instruct" is a 1.7 billion parameter model that
# already understands language. We're fine-tuning it (not training from scratch).
# Unsloth provides optimized versions that load and train faster.
MODEL_NAME = "unsloth/SmolLM2-1.7B-Instruct"

# MAX_SEQ_LENGTH: Maximum number of tokens (roughly words/subwords) allowed in
# a single training example. If a conversation has more tokens than this, the
# end gets cut off. 2048 is plenty for short Q&A pairs (a typical pair is ~100-300
# tokens). Setting this too high wastes VRAM; too low cuts off long answers.
MAX_SEQ_LENGTH = 2048

# LOAD_IN_4BIT: If True, the base model's weights are compressed from 16-bit
# (2 bytes per weight) to 4-bit (0.5 bytes per weight). This is the "Q" in QLoRA.
# Effect: VRAM drops from ~3.4 GB to ~1 GB for the base model, at a small cost
# in precision. The LoRA adapters are still trained in full 16-bit precision.
#
# IMPORTANT FOR OLLAMA EXPORT: When set to True (4-bit), the merge step must
# dequantize 4-bit → 16-bit, which introduces error. The LoRA corrections were
# learned against the exact 4-bit values, but after dequantization those values
# shift slightly, so the corrections no longer line up perfectly. This causes
# quality loss when exporting to Ollama (double quantization: NF4→bf16→q8_0).
#
# When set to False (16-bit), the base weights are exact during training AND
# after merge. Only Ollama's q8_0 quantization introduces rounding — one round
# of error instead of two. Requires more VRAM (~12 GB total) but RTX 4090 handles it.
LOAD_IN_4BIT = False


# --- LoRA hyperparameters ---------------------------------------------------
# LoRA = "Low-Rank Adaptation". Instead of changing the model's original weight
# matrices (which are huge), LoRA inserts two small matrices (A and B) next to
# each target layer. During training, only A and B are updated. The original
# weights stay frozen. After training, A×B produces a small "correction" that
# gets added to the original weights.

# LORA_R (rank): The inner dimension of the A and B matrices.
# Think of it as "how much new knowledge can the adapter store."
#   - r=16: 16 dimensions → smaller, faster, less capacity (good for small data).
#   - r=64: 64 dimensions → bigger, slower, more capacity (better for larger data).
# With 69 training examples, r=16 is sufficient. Too high risks memorizing noise.
LORA_R = 64

# LORA_ALPHA: Controls how strongly the LoRA correction affects the model.
# The actual scaling factor is (alpha / r). With alpha=16 and r=16, the scale
# is 1.0, meaning the LoRA correction is applied at full strength.
# If alpha=32 and r=16, scale=2.0 → corrections are doubled (more aggressive).
# Rule of thumb: set alpha = r for a balanced starting point.
LORA_ALPHA = 64

# LORA_DROPOUT/bye: Probability of randomly zeroing out LoRA activations during
# training. Dropout is a regularization technique that prevents overfitting
# by forcing the model to not rely too heavily on any single connection.
# 0 = no dropout. Unsloth uses an optimized code path when dropout=0.
# For very small datasets, you might try 0.05 or 0.1 if overfitting is severe.
LORA_DROPOUT = 0

# TARGET_MODULES: Which layers inside each transformer block get LoRA adapters.
# Each transformer block contains two main components:
#   1. Self-Attention: has q_proj (Query), k_proj (Key), v_proj (Value), o_proj (Output)
#      - These help the model decide which words to "pay attention to" when generating.
#   2. FFN (Feed-Forward Network): has gate_proj, up_proj, down_proj
#      - These store factual knowledge and do the "thinking" after attention.
# By adding LoRA to ALL of them, the model can learn both new attention patterns
# (how to connect "Abdurrahman" with "NUAA") and new facts (the actual information).
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# --- Training hyperparameters -----------------------------------------------

# NUM_TRAIN_EPOCHS: One epoch = the model sees every training example once.
# With 69 examples, 1 epoch = 69 examples processed. With 50 epochs, the model
# sees each example 50 times. This sounds like a lot, but LLMs need repetition
# to memorize facts. We set this high because early stopping will automatically
# terminate training when overfitting is detected (usually around epoch 10-15).
NUM_TRAIN_EPOCHS = 1000

# LEARNING_RATE: How much the weights change after each training step.
# Think of it as "step size" when walking toward the answer:
#   - Too high (e.g., 1e-3): takes big steps, overshoots, training is unstable.
#   - Too low (e.g., 1e-6): takes tiny steps, never reaches the answer in time.
#   - 1e-4 (0.0001): a proven sweet spot for QLoRA fine-tuning.
LEARNING_RATE = 1e-4

# PER_DEVICE_TRAIN_BATCH_SIZE: How many training examples the GPU processes
# simultaneously in one forward pass. Larger batch = more stable gradients
# but more VRAM. With short Q&A pairs and 4-bit loading, batch_size=2 fits
# easily in an RTX 4090's 24 GB.
PER_DEVICE_TRAIN_BATCH_SIZE = 2

# GRADIENT_ACCUMULATION_STEPS: Instead of updating weights after every 2
# examples (batch_size=2), accumulate gradients over 4 mini-batches first,
# then update. This simulates a larger "effective batch size" of 2×4=8 without
# needing the VRAM for 8 examples at once.
# Larger effective batch = smoother, more stable training.
GRADIENT_ACCUMULATION_STEPS = 4

# WARMUP_STEPS: For the first N steps, the learning rate gradually increases
# from 0 to LEARNING_RATE. This prevents the model from making wild, destructive
# updates at the very beginning when the LoRA adapters are randomly initialized.
# After warmup, the learning rate follows the cosine schedule (see lr_scheduler_type).
WARMUP_STEPS = 10

# LOGGING_STEPS: Print the training loss to the console every N steps.
# 1 = print after every single step (maximum visibility into training progress).
LOGGING_STEPS = 1


# --- Validation / overfitting detection / early stopping --------------------

# EVAL_STEPS: Run the validation set through the model every N training steps
# and compute the validation loss. 1 = evaluate after every step. This gives
# the most detailed view of when overfitting starts, at a tiny speed cost
# (~0.1 seconds per eval with 12 validation examples).
EVAL_STEPS = 1

# EARLY_STOP_MIN_STEPS: Don't check for early stopping until this many steps
# have passed. Why? In the first ~80 steps, the LoRA adapters transition from
# random initialization to useful weights. During this phase, the validation
# loss can spike temporarily (it always recovers). Without this guard, early
# stopping would kill training during a normal learning spike.
EARLY_STOP_MIN_STEPS = 80

# EARLY_STOP_THRESHOLD: The percentage above the best validation loss that
# triggers overfitting detection. 0.20 = 20% above best.
# Example: if best val_loss was 0.47, training continues as long as val_loss
# stays below 0.47 × 1.20 = 0.564. Once it exceeds that for PATIENCE
# consecutive evals, training stops.
EARLY_STOP_THRESHOLD = 0.30

# EARLY_STOP_PATIENCE: How many consecutive evaluations the val_loss must
# stay above the threshold before training actually stops.
# This prevents stopping on a single random spike. With patience=10,
# the model must be consistently overfitting for 10 straight evaluations.
EARLY_STOP_PATIENCE = 10

# EVAL_STRATEGY: When to run evaluation. "steps" means every EVAL_STEPS steps.
# Alternative: "epoch" would run evaluation once per full pass through the data.
EVAL_STRATEGY = "steps"

# LOAD_BEST_MODEL_AT_END: If True, after training finishes, the trainer would
# roll back to the checkpoint with the lowest validation loss. We set False
# because our early stopping strategy deliberately allows training up to 20%
# above the best val loss — the model at the stop point has seen more data
# and often gives better answers than the "best" checkpoint.
LOAD_BEST_MODEL_AT_END = False

# METRIC_FOR_BEST_MODEL: Which number to use when deciding which checkpoint
# is "best". "eval_loss" = the loss computed on the validation set.
# Lower eval_loss = the model predicts the validation answers more confidently.
METRIC_FOR_BEST_MODEL = "eval_loss"

# SAVE_STEPS: Save a full checkpoint to disk every N steps. Checkpoints are
# large (~250 MB each), so we don't save every step. Every 10 steps is enough
# to have a recent checkpoint to roll back to if needed.
SAVE_STEPS = 10

# SAVE_TOTAL_LIMIT: Keep at most this many checkpoints on disk. When a new
# checkpoint is saved and the limit is exceeded, the oldest one is deleted.
# This prevents filling up your hard drive during long training runs.
SAVE_TOTAL_LIMIT = 3

# SEED: Random seed for reproducibility. Setting a fixed seed means that
# if you run the script twice with the same data and settings, you get
# the same results. 42 is a common default (a reference to The Hitchhiker's
# Guide to the Galaxy).
SEED = 42


# =============================================================================
# EARLY STOPPING CALLBACK
# =============================================================================

class EarlyStopOnValIncrease(TrainerCallback):
    """A custom callback that monitors validation loss during training and
    stops training early when the model starts overfitting.

    How it works:
    1. After each validation evaluation, it checks the validation loss.
    2. If the loss is a new best (lowest ever), it resets the counter.
    3. If the loss exceeds (best_loss × (1 + threshold)) AND we're past
       min_steps, it starts counting consecutive bad evaluations.
    4. If the count reaches `patience`, it tells the trainer to stop.

    This is like a safety net: even though we set 50 epochs, training will
    automatically stop when it's no longer helping and starting to hurt.
    """

    def __init__(self, threshold=0.10, patience=10, min_steps=80):
        self.threshold = threshold       # How much above best is "too much" (0.20 = 20%)
        self.patience = patience         # How many bad evals in a row before stopping
        self.min_steps = min_steps       # Don't even check until this many steps
        self.best_val_loss = float("inf")  # Best (lowest) val loss seen so far
        self.consecutive_above = 0       # Counter of consecutive bad evaluations

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called automatically by the trainer after each evaluation run."""
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        # Case 1: New best! Reset the counter.
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.consecutive_above = 0

        # Case 2: Val loss is above threshold AND we're past the safe zone.
        elif state.global_step >= self.min_steps and val_loss > self.best_val_loss * (1 + self.threshold):
            self.consecutive_above += 1
            if self.consecutive_above >= self.patience:
                print(
                    f"\n  >>> EARLY STOP at step {state.global_step}: "
                    f"val_loss {val_loss:.4f} exceeded best "
                    f"{self.best_val_loss:.4f} by >{self.threshold*100:.0f}% "
                    f"for {self.patience} consecutive evals."
                )
                # This flag tells the Hugging Face Trainer to stop the training loop.
                control.should_training_stop = True

        # Case 3: Val loss is above best but not above threshold — reset counter.
        else:
            self.consecutive_above = 0


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset_from_json(path, tokenizer):
    """Load a JSON file of conversations and convert each one into a single
    text string that the model can learn from.

    The JSON file looks like:
      [{"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}]

    The tokenizer's chat template converts each conversation into a formatted
    string like:
      <|im_start|>user\nWho is X?<|im_end|>\n<|im_start|>assistant\nX is...<|im_end|>

    This formatted string is what the model actually trains on. The model learns
    to predict each next token in the assistant's response, given the user's question.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)  # Load the entire JSON array into memory

    texts = []
    for item in raw:
        messages = item["messages"]  # Extract the conversation messages
        # apply_chat_template: converts the message list into a single formatted string
        # tokenize=False: return a string (not token IDs) — the trainer tokenizes later
        # add_generation_prompt=False: don't add the assistant prompt at the end,
        #   because the assistant's response is already included in the training data
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Dataset.from_dict: creates a Hugging Face Dataset object from a dictionary.
    # The trainer expects a dataset with a "text" column containing the training strings.
    return Dataset.from_dict({"text": texts})


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    # --- Safety check: training requires a CUDA-capable GPU (NVIDIA) ---
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA GPU.")

    # =========================================================================
    # STEP 1: Load the pre-trained base model and its tokenizer
    # =========================================================================
    # The model contains 1.7 billion parameters that already understand language.
    # The tokenizer converts text to numbers (token IDs) and back.
    # We load the model in 4-bit precision (LOAD_IN_4BIT=True) to save VRAM.
    # dtype=None lets Unsloth auto-detect the best floating point format (bfloat16).
    print("=" * 60)
    print("STEP 1: Loading base model and tokenizer")
    print("=" * 60)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # =========================================================================
    # STEP 2: Attach LoRA adapters to the model
    # =========================================================================
    # This is where QLoRA happens. We take the frozen 4-bit base model and
    # attach small trainable adapter matrices (LoRA) to each target layer.
    # Only these adapters will be updated during training — the base model
    # stays completely unchanged.
    #
    # bias="none": don't add bias terms to LoRA layers (saves parameters).
    # use_gradient_checkpointing="unsloth": a memory optimization that
    #   recomputes some intermediate values instead of storing them all in VRAM.
    #   Trades a little speed for significant VRAM savings.
    # random_state=SEED: ensures the random initialization of LoRA matrices
    #   is reproducible.
    print("\n" + "=" * 60)
    print("STEP 2: Adding LoRA adapters")
    print("=" * 60)
    print(f"  LoRA rank (r): {LORA_R}")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Target modules: {TARGET_MODULES}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # Count parameters to show how efficient LoRA is.
    # "Trainable" = only the LoRA adapter weights (the ones that change).
    # "Total" = all weights including the frozen base model.
    # Typically, LoRA trains only 4-7% of the total parameters.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,}  ({100 * trainable / total:.2f}% of {total:,})")

    # =========================================================================
    # STEP 3: Load the training and validation datasets
    # =========================================================================
    # Training data: personal Q&A pairs. The model learns from ALL of these.
    # Validation data: general knowledge Q&A. The model is NEVER trained on
    #   these — they only measure whether the model is losing general ability.
    print("\n" + "=" * 60)
    print("STEP 3: Loading datasets (separate train & validation files)")
    print("=" * 60)
    print(f"  Train file: {DATASET_PATH}")
    print(f"  Val file:   {VAL_DATASET_PATH}")

    train_dataset = load_dataset_from_json(DATASET_PATH, tokenizer)
    val_dataset = load_dataset_from_json(VAL_DATASET_PATH, tokenizer)

    print(f"  Train examples: {len(train_dataset)}  (100% of training data)")
    print(f"  Val examples:   {len(val_dataset)}  (hand-crafted, never trained on)")
    print(f"  First train example (preview):")
    preview = train_dataset[0]["text"]
    if len(preview) > 300:
        preview = preview[:300] + "..."
    print(f"  {preview}")

    # =========================================================================
    # STEP 4: Configure and run training
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Starting training (with validation monitoring)")
    print("=" * 60)
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Eval every: {EVAL_STEPS} steps")
    print(f"  Early stop: after step {EARLY_STOP_MIN_STEPS}, if val_loss > best * "
          f"{1 + EARLY_STOP_THRESHOLD:.2f} for {EARLY_STOP_PATIENCE} consecutive evals")
    print(f"  Save model from: {'best val checkpoint' if LOAD_BEST_MODEL_AT_END else 'early stop point'}")
    print()
    print("  Watch the output below:")
    print("    'loss'      = training loss (should decrease)")
    print("    'eval_loss' = validation loss (if this goes UP, overfitting!)")
    print()

    # Create our custom early stopping callback
    early_stop_cb = EarlyStopOnValIncrease(
        threshold=EARLY_STOP_THRESHOLD,
        patience=EARLY_STOP_PATIENCE,
        min_steps=EARLY_STOP_MIN_STEPS,
    )

    # SFTTrainer: the engine that runs the training loop.
    # It handles: batching data → forward pass → loss calculation →
    #   backpropagation → weight update → logging → evaluation → checkpointing.
    trainer = SFTTrainer(
        model=model,             # The model with LoRA adapters attached
        tokenizer=tokenizer,     # Needed to tokenize the training text
        train_dataset=train_dataset,   # What the model learns from
        eval_dataset=val_dataset,      # What we test on (never trained on)
        callbacks=[early_stop_cb],     # Our custom early stopping logic

        args=SFTConfig(
            # --- Where to save checkpoints ---
            output_dir=OUTPUT_DIR,

            # --- Batch size settings ---
            # These three together determine effective batch size = 2 × 4 = 8
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

            # --- Learning schedule ---
            warmup_steps=WARMUP_STEPS,       # Gradual LR ramp-up at the start
            num_train_epochs=NUM_TRAIN_EPOCHS,  # Max passes through the data
            learning_rate=LEARNING_RATE,      # Step size for weight updates

            # --- Logging ---
            logging_steps=LOGGING_STEPS,      # Print loss every N steps
            seed=SEED,

            # --- Tokenization ---
            max_seq_length=MAX_SEQ_LENGTH,    # Truncate examples longer than this

            # --- Floating point precision ---
            # fp16: use 16-bit float (for older GPUs that don't support bfloat16)
            # bf16: use bfloat16 (preferred on modern GPUs like RTX 3090/4090)
            # Exactly one of these will be True, the other False.
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            # --- Optimizer ---
            # adamw_8bit: AdamW optimizer in 8-bit precision. AdamW is the standard
            # optimizer for training transformers. The 8-bit version uses less VRAM
            # for optimizer states (momentum, variance) with minimal quality loss.
            optim="adamw_8bit",

            # weight_decay: L2 regularization — penalizes large weights to prevent
            # overfitting. 0.01 is a gentle amount. It nudges the model toward
            # simpler solutions that generalize better.
            weight_decay=0.001,

            # lr_scheduler_type: How the learning rate changes over time.
            # "cosine" means: after warmup, the LR follows a cosine curve from
            # LEARNING_RATE down to ~0. This gives a smooth, gradual slowdown
            # that helps the model settle into a good solution near the end.
            lr_scheduler_type="cosine",

            # dataset_text_field: tells the trainer which column in the dataset
            # contains the text to train on. Our dataset has a "text" column.
            dataset_text_field="text",

            # --- Evaluation settings ---
            eval_strategy=EVAL_STRATEGY,   # "steps" = eval every N steps
            eval_steps=EVAL_STEPS,         # N = 1 (eval every step)

            # --- Checkpointing ---
            save_steps=SAVE_STEPS,              # Save checkpoint every 10 steps
            save_total_limit=SAVE_TOTAL_LIMIT,  # Keep max 3 checkpoints on disk

            # --- Best model selection ---
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,  # False: use stop point
            metric_for_best_model=METRIC_FOR_BEST_MODEL,    # "eval_loss"
            greater_is_better=False,  # For loss, lower is better (not higher)
        ),
    )

    # trainer.train() — THIS IS WHERE THE ACTUAL TRAINING HAPPENS.
    # The trainer loops through the data for up to NUM_TRAIN_EPOCHS passes:
    #   1. Takes a batch of training examples
    #   2. Feeds them through the model (forward pass)
    #   3. Computes how wrong the model's predictions are (loss)
    #   4. Calculates how to fix each weight (backpropagation)
    #   5. Updates the LoRA adapter weights (optimizer step)
    #   6. Every EVAL_STEPS, runs validation and checks early stopping
    # This continues until either all epochs are done or early stopping triggers.
    trainer.train()

    # =========================================================================
    # TRAINING SUMMARY: Print a table showing train vs val loss at each step
    # =========================================================================
    # This table helps you visually identify:
    #   - Where the model learned the most (both losses dropping)
    #   - Where the best checkpoint was (lowest val loss)
    #   - Where overfitting started (train loss dropping, val loss rising)
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY: Train vs Validation Loss")
    print("=" * 60)
    print(f"  {'Step':<8} {'Train Loss':>12} {'Val Loss':>12} {'Status':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    # Collect all evaluation logs from the training history
    eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]

    # First pass: find the step with the absolute best (lowest) val loss
    best_val_loss = float("inf")
    best_val_step = 0
    for log in eval_logs:
        if log["eval_loss"] < best_val_loss:
            best_val_loss = log["eval_loss"]
            best_val_step = log.get("step", 0)

    # Second pass: print each step with a status label
    # Before the best step: blank (still learning, can't be overfitting)
    # At the best step: "<<< BEST"
    # After the best step: compare to best val loss for overfitting detection
    past_best = False
    for log in eval_logs:
        step = log.get("step", "?")
        val_loss = log["eval_loss"]

        # Find the matching training loss for this step
        train_log = None
        for tl in trainer.state.log_history:
            if "loss" in tl and tl.get("step") == step:
                train_log = tl
                break
        train_loss = train_log["loss"] if train_log else float("nan")

        # Determine status label
        if step == best_val_step:
            status = "<<< BEST"
            past_best = True
        elif not past_best:
            status = ""        # Before best: still learning
        elif val_loss > best_val_loss * 1.10:
            status = "OVERFIT"   # >10% above best: clearly overfitting
        elif val_loss > best_val_loss * 1.05:
            status = "OVERFIT?"  # 5-10% above best: starting to drift
        else:
            status = "OK"        # Within 5% of best: still healthy

        print(f"  {step:<8} {train_loss:>12.4f} {val_loss:>12.4f} {status:>12}")

    # Print final summary: where the best was and where we stopped
    if best_val_loss < float("inf"):
        print(f"\n  Best val loss: {best_val_loss:.4f} at step {best_val_step}")
        stopped_step = trainer.state.global_step
        stopped_val = eval_logs[-1]["eval_loss"] if eval_logs else float("nan")
        print(f"  Stopped at step {stopped_step} (val loss: {stopped_val:.4f})")
        if LOAD_BEST_MODEL_AT_END:
            print(f"  Model loaded from BEST checkpoint (step {best_val_step})")
        else:
            print(f"  Model saved from STOP point (step {stopped_step})")

    # =========================================================================
    # STEP 5: Save the trained LoRA adapters to disk
    # =========================================================================
    # model.save_pretrained: saves ONLY the LoRA adapter weights (not the full
    #   base model). The output is a small folder (~250 MB) containing:
    #   - adapter_model.safetensors: the learned LoRA weight matrices
    #   - adapter_config.json: LoRA settings (rank, alpha, target modules)
    # tokenizer.save_pretrained: saves the tokenizer files so you can reload
    #   everything together later without re-downloading.
    print("\n" + "=" * 60)
    print("STEP 5: Saving LoRA adapters")
    print("=" * 60)
    print(f"  Saving to: {OUTPUT_DIR}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"  Done! Adapter files saved in {OUTPUT_DIR}")
    print(f"\n  Next: run test_finetuned.py to test your fine-tuned model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
