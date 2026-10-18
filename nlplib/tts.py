import json
import torch
from TTS.api import TTS
from nlplib.config import MODEL_NAME, OUTPUT_DIR
from nlplib.write import write_file

def load_yourtts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_file(f"Using device: {device}")
    
    return TTS(model_name=MODEL_NAME).to(device)

def get_supported_languages(tts):
    return list(tts.languages or [])

def output_language_report():
    tts = load_yourtts()
    languages = get_supported_languages(tts)

    report = {
        "model": MODEL_NAME,
        "languages": languages,
        "lithuanian_available": any(
            language.lower() in ("lt", "lt-lt", "lithuanian")
            for language in languages
        ),
    }

    write_file(json.dumps(report, indent=2, ensure_ascii=False))

def test_speech(file_name: str, output_file_name: str = "test", file_format: str = "wav"):
    tts = load_yourtts()

    tts.tts_to_file(
        text = "This is a test of the speech synthesis system",
        speaker_wav = str(OUTPUT_DIR / f"{file_name}.{file_format}"),
        language = "en",
        file_path = str(OUTPUT_DIR / f"{output_file_name}.{file_format}"),
    )

    write_file(f"Synthesized speech saved to {output_file_name}.{file_format}")