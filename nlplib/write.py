from nlplib.config import OUTPUT_DIR

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
output_path = OUTPUT_DIR / "log.txt"

def clear_file() -> None:
    with open(output_path, 'w', encoding = "utf-8") as f:
        f.write('')

def write_file(text: str) -> None:
    with open(output_path, 'a', encoding = "utf-8") as f:
        f.write(f"{text}\n")

def print_audio_info(file_path, audio, sample_rate, info):
    duration = len(audio) / sample_rate

    write_file(f"\nFile: {file_path.name}")
    write_file(f"Duration: {duration:.2f} seconds")
    write_file(f"Sample rate: {sample_rate} Hz")
    write_file(f"Channels: {info.channels}")
    write_file(f"Format: {info.format}")
    write_file(f"Subtype / bit depth: {info.subtype}")

    return duration