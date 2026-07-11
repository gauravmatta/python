import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Cast the large transformer to float16 to halve VRAM usage
model.t3 = model.t3.half()
model.s3gen = model.s3gen.half()

with open("sample/text/readh.txt", "r") as f:
    text = f.read()

with torch.autocast("cuda", dtype=torch.float16):
    wav = model.generate(
        text,
        language_id="hi",
        audio_prompt_path="sample/voice/sample.wav",
    )

torchaudio.save("output.wav", wav, model.sr)
