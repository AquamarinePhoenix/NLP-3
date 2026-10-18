import subprocess
from pathlib import Path
from nlplib.write import write_file
from nlplib.config import INPUT_DIR, OUTPUT_DIR

def convert_to_wav(file_name: str, file_format: str, output_format = "wav"):
    input_path = Path(INPUT_DIR) / f"{file_name}.{file_format}"
    output_path = Path(OUTPUT_DIR) / f"{file_name}.{output_format}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "22050",
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    write_file(
        f"Converted successfully to file {output_path.name} in {OUTPUT_DIR}"
    )