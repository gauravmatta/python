import whisper
from pathlib import Path
from deep_translator import GoogleTranslator

# Batched translation: one request per chunk; newline = one cue (deep_translator limit ~5000 chars)
TRANSLATE_CHAR_LIMIT = 5000
CHUNK_MAX_LINES = 25

# Load the Whisper base model
model = whisper.load_model("base")

# Define the videos folder path
videos_folder = "videos"

# Check if the videos folder exists
if not Path(videos_folder).exists():
    raise FileNotFoundError(f"Videos folder not found: {videos_folder}")

# Create output folders if they don't exist
output_folder_path = Path("subtitles")
output_folder_path.mkdir(parents=True, exist_ok=True)
english_folder_path = output_folder_path / "english"
english_folder_path.mkdir(parents=True, exist_ok=True)

# Find all MP4 video files in the videos folder
video_files = list(Path(videos_folder).glob("*.mp4"))

if not video_files:
    print(f"No MP4 video files found in: {videos_folder}")
else:
    supported_languages = GoogleTranslator().get_supported_languages()
    translation_targets = sorted(
        lang for lang in supported_languages if lang.lower() != "english"
    )

    # Process each video file
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        # Check if the file exists
        if not video_path.exists():
            print(f"  Warning: File not found: {video_path}")
            continue

        video_name = video_path.stem
        english_output_file = english_folder_path / f"{video_name}_english.vtt"

        def all_subtitles_already_exist():
            if not english_output_file.is_file():
                return False
            for lang in translation_targets:
                p = output_folder_path / lang / f"{video_name}_{lang}.vtt"
                if not p.is_file():
                    return False
            return True

        if all_subtitles_already_exist():
            print(
                "  Skipping — English and all translated languages already on disk "
                f"({len(translation_targets)} languages)."
            )
            continue

        if english_output_file.is_file():
            print(f"  Resuming: using existing English VTT ({english_output_file.name})")
            content = english_output_file.read_text(encoding="utf-8")
        else:
            result = model.transcribe(str(video_path))

            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds - int(seconds)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

            lines = ["WEBVTT", ""]
            for segment in result["segments"]:
                start = format_time(segment["start"])
                end = format_time(segment["end"])
                text = segment["text"].strip()
                lines.append(f"{start} --> {end}")
                lines.append(text)
                lines.append("")

            content = "\n".join(lines)
            english_output_file.write_text(content, encoding="utf-8")
            print(f"  English subtitles saved to: {english_output_file}")

        subtitle_blocks = content.split("\n\n")

        # (timestamp line, cue text) — one text line per cue for batch join/split
        cue_pairs = []
        for block in subtitle_blocks:
            if not block.strip() or block.strip().startswith("WEBVTT"):
                continue
            block_lines = block.strip().split("\n")
            if not block_lines or "-->" not in block_lines[0]:
                continue
            ts_line = block_lines[0].strip()
            body = " ".join(line.strip() for line in block_lines[1:] if line.strip())
            cue_pairs.append((ts_line, body))

        # Translate only missing language files (resume-friendly)
        for lang in translation_targets:
            output_file = output_folder_path / lang / f"{video_name}_{lang}.vtt"

            if output_file.is_file():
                print(f"  Skipping {lang.capitalize()} (already exists): {output_file}")
                continue

            try:
                translator = GoogleTranslator(source="en", target=lang)

                lang_folder_path = output_folder_path / lang
                lang_folder_path.mkdir(parents=True, exist_ok=True)

                # Create target language subtitle blocks (batched translate per chunk)
                target_lines = []
                target_lines.append("WEBVTT")
                target_lines.append("")

                translated_texts = []
                idx = 0
                n_cues = len(cue_pairs)
                while idx < n_cues:
                    end = min(idx + CHUNK_MAX_LINES, n_cues)
                    texts = [cue_pairs[i][1] for i in range(idx, end)]
                    blob = "\n".join(texts)
                    while len(blob) > TRANSLATE_CHAR_LIMIT and end > idx + 1:
                        end -= 1
                        texts = [cue_pairs[i][1] for i in range(idx, end)]
                        blob = "\n".join(texts)
                    if end == idx:
                        end = idx + 1
                        texts = [cue_pairs[i][1] for i in range(idx, end)]
                        blob = "\n".join(texts)

                    try:
                        out_blob = translator.translate(blob)
                    except Exception as e:
                        print(f"  Error translating chunk to {lang}: {e}")
                        out_blob = None

                    parts = []
                    if out_blob is not None:
                        parts = out_blob.replace("\r\n", "\n").split("\n")

                    if out_blob is None or len(parts) != len(texts):
                        parts = []
                        for t in texts:
                            try:
                                parts.append(translator.translate(t))
                            except Exception as e2:
                                print(f"  Error translating line to {lang}: {e2}")
                                parts.append(t)

                    translated_texts.extend(parts)
                    idx = end

                for (ts_line, _), txt in zip(cue_pairs, translated_texts):
                    target_lines.append(ts_line)
                    target_lines.append(txt)
                    target_lines.append("")
                
                # Write target language subtitles to file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(target_lines))
                
                print(f"  {lang.capitalize()} subtitles saved to: {output_file}")
                
            except Exception as e:
                print(f"  Error with language {lang}: {e}")
                continue

print(f"\nTotal videos processed: {len(video_files)}")