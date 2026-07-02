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
# python main.py
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

# Use a pipeline as a high-level helper
# The 'pipeline' abstraction handles downloading, tokenizing, and running the model for you.
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-230M")
messages = [
    {"role": "scientist", "content": "Why mud is brown?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))