import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
# No manual .half() calls on submodules — let autocast manage precision

with open("sample/text/readh.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
sentences = [s for s in sentences if s]

wavs = []
for i, sentence in enumerate(sentences):
    print(f"Generating {i+1}/{len(sentences)}: {sentence[:40]}...")
    torch.cuda.empty_cache()
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            wav = model.generate(
                sentence,
                language_id="hi",
                audio_prompt_path="sample/voice/samplea.wav",
            )
        wavs.append(wav)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM on sentence {i+1}, skipping")
        torch.cuda.empty_cache()
        continue

final_wav = torch.cat(wavs, dim=-1)
print("Model sample rate:", model.sr)
torchaudio.save("output.wav", final_wav, model.sr)