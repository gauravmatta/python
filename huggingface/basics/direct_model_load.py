# Title: Use HuggingFace Models Locally
#
# Description:
# This script demonstrates how to run a model from Hugging Face locally on your machine.
# Unlike the previous API methods (which run in the cloud), this method downloads
# the model weights to your computer and runs them using your own CPU/GPU.
#
# This uses the `transformers` library, which is the standard for running Hugging Face models.
#
# Installation:
# pip install transformers torch
# (You might also need `accelerate` or `bitsandbytes` depending on the model)
#
# How to run:
# python direct_model_load.py
#
# --- IMPORTANT: MANAGING DISK SPACE ---
# Hugging Face models are LARGE (Gigabytes). They are stored in a cache folder.
# Default Cache Location:
# - Windows: C:\Users\<Username>\.cache\huggingface\hub
# - Mac/Linux: ~/.cache/huggingface/hub
#
# How to DELETE a model:
# 1. Option A (Manual): Go to the cache folder above and delete the folders named "models--..."
# 2. Option B (CLI): Install the CLI tool: `pip install huggingface_hub[cli]`
#    Then run: `hf cache list` (to see models)
#    Then run: `hf cache delete` (to interactively delete them)

from transformers import pipeline, GenerationConfig

# Loading model directly
# Initialize the pipeline.
# task="text-generation": Tells the library what kind of job we want to do.
# model="LiquidAI/LFM2.5-1.2B-Thinking": The specific model ID on Hugging Face Hub.
#
# Note: The first time you run this, it will DOWNLOAD the model (could be several GBs).
# Subsequent runs will use the cached version.
# pipe = pipeline("text-generation", model="LiquidAI/LFM2.5-1.2B-Thinking")
pipe = pipeline("text-generation", model="LiquidAI/LFM2.5-230M")

# Define the input prompt.
# Many modern models expect a list of messages (Chat format).
messages = [
    {"role": "user", "content": "Why is the sky blue?"},
]

# Run the model with CUSTOM PARAMETERS.
#
# Arguments Explained:
# 1. max_new_tokens (int): The maximum number of *new* words the model can generate.
#    Increasing this prevents the answer from being cut off mid-sentence.
#    Default is often low (50-100).
#
# 2. do_sample (bool):
#    - True: The model picks the next word randomly based on probabilities (Creative/Varied).
#    - False: The model always picks the single most likely next word (Deterministic/Repetitive).
#
# 3. temperature (float, 0.0 - 1.0+): Controls "Randomness" or "Creativity".
#    - Low (e.g., 0.1): Predictable, focused, factual. Good for coding or math.
#    - High (e.g., 0.9): Creative, diverse, sometimes chaotic. Good for poetry or brainstorming.
#    - Note: Only works if do_sample=True.
#
# 4. top_k (int): Limits the random choice to the top K most likely words.
#    - e.g., top_k=50 means "Only consider the 50 best words for the next step".
#    - Helps prevent the model from choosing very weird/nonsense words.
#
# 5. top_p (float, 0.0 - 1.0): Nucleus Sampling.
#    - e.g., top_p=0.9 means "Consider the smallest set of words whose probabilities add up to 90%".
#    - A more dynamic version of top_k.

generation_config = GenerationConfig(
    max_new_tokens=5024,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.5
)

output = pipe(
    messages,
    clean_up_tokenization_spaces=False,
    generation_config=generation_config
)

# The result is a complex list structure. We access [0]['generated_text'] to get the answer.
# Note: The indexing [1]['content'] is specific to how this model returns the chat history.
print(output[0]['generated_text'][1]['content'])
# print(output[0]['generated_text'][1]['content'].split("</think>")[1])