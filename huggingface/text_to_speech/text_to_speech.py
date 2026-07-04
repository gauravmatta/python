# Title: Text-to-Speech with Qwen TTS
#
# Description:
# This script demonstrates how to use a specialized Text-to-Speech (TTS) model.
# We are using "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".
#
# Unlike generic models, some specialized models require their own libraries (like 'qwen_tts')
# or specific configurations.
#
# Troubleshooting:
# If you see errors about "FlashAttention", it means your computer is missing a specific
# optimization library that is hard to install on Windows. We disable it below to ensure compatibility.
#
# Installation:
# pip install torch soundfile transformers accelerate qwen_tts
#
# How to run:
# python text_to_spech.py

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Initialize the model.
# .from_pretrained() downloads the model weights from Hugging Face.
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    # device_map="auto": Automatically puts the model on GPU if available, otherwise CPU.
    device_map="auto",
    # dtype=torch.bfloat16: Uses Brain Floating Point format (good for newer GPUs).
    # If you get an error here on older GPUs, change it to torch.float16 or torch.float32.
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", # <--- REMOVED/COMMENTED OUT
    # This should be removed on Windows because 'flash_attn' is hard to install on Windows.
    # The model will now use the standard, compatible attention mechanism.
    # flash-attn has no wheels for Python 3.14 and won't build from source with Python 3.14
)


# --- Helper Function: List Capabilities ---
def print_model_capabilities(model):
    print("\n--- Model Capabilities ---")

    # Get supported languages
    # The model wrapper usually has a method or property for this.
    # Based on the library code, it's get_supported_languages()
    languages = model.get_supported_languages()
    if languages:
        print(f"Supported Languages ({len(languages)}):")
        print(", ".join(languages))
    else:
        print("Supported Languages: Not explicitly listed by model (likely Auto-detect).")

    # Get supported speakers
    # Based on the library code, it's get_supported_speakers()
    speakers = model.get_supported_speakers()
    if speakers:
        print(f"\nSupported Speakers ({len(speakers)}):")
        print(", ".join(speakers))
    else:
        print("\nSupported Speakers: Not explicitly listed by model.")
    print("--------------------------\n")


# Call the function to see what we can use
print_model_capabilities(model)

# --- Single Inference ---
# Generating audio for a single sentence.
print("Generating single audio...")
wavs, sr = model.generate_custom_voice(
    text="По нaбережной Москвы иду к пaрку Горького Слушaю я ветер перемен",
    language="Russian",  # 'Auto' detects language, or specify 'Chinese'/'English' etc.
    speaker="Sohee",  # The name of the voice profile to use
    instruct="сказал крайне сердитым тоном",  # Instruction: "Speak in a very angry tone"
)
# Save the result to a WAV file
sf.write("output_custom_voice.wav", wavs[0], sr)
print("Saved output_custom_voice.wav")

# --- Batch Inference ---
# Generating audio for multiple sentences at once (faster than loop).
print("Generating batch audio...")
wavs, sr = model.generate_custom_voice(
    text=[
        "На самом деле я понял, что у меня особенно хорошо получается наблюдать за эмоциями других людей.",
        "She said she would be here by noon."
    ],
    language=["Russian", "English"],
    speaker=["Uncle_fu", "Vivian"],
    instruct=["сказал чрезвычайно радостным тоном", "Very happy."]  # First has no instruction, second is "Very happy"
)

# Save the batch results
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
print("Saved batch outputs.")

# --- Advanced Usage: Controlling Style & Randomness ---
# Just like text LLMs, TTS models have parameters to control "creativity" or stability.
#
# Key Parameters:
# 1. temperature (default 0.9):
#    - Lower (e.g., 0.1): More stable, robotic, consistent pronunciation.
#    - Higher (e.g., 1.0+): More expressive, emotional, but potentially unstable (mumbling).
#
# 2. top_p (default 1.0):
#    - Controls the "nucleus sampling". Reducing it (e.g., 0.8) can make speech clearer
#      by ignoring unlikely acoustic tokens.
#
# 3. repetition_penalty (default 1.05):
#    - Prevents the model from getting stuck in a loop (e.g., "Hello hello hello...").
#    - Increase slightly (e.g., 1.1 or 1.2) if you hear stuttering.

print("Generating advanced audio with custom parameters...")
wavs, sr = model.generate_custom_voice(
    text="This is a test of the advanced generation parameters. I should sound very stable.",
    language="English",
    speaker="Ryan",
    instruct="Say it in a very scared and shocked tone.",

    # Advanced Params passed as kwargs:
    temperature=0.7,  # Slightly lower for stability
    top_p=0.5,  # Filter out low-probability sounds
    repetition_penalty=1.1,  # Prevent stuttering
    max_new_tokens=4096  # Allow for longer speech generation if needed
)

sf.write("output_advanced.wav", wavs[0], sr)
print("Saved output_advanced.wav")