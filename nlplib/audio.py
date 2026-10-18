from pathlib import Path
import numpy as np
import soundfile as sf
from nlplib.config import OUTPUT_DIR
from nlplib.write import write_file, print_audio_info
from nlplib.plot import plot_waveforms, plot_f0, plot_spectra

def load_audio(file_path):
    audio, sample_rate = sf.read(file_path, dtype = "float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    info = sf.info(file_path)

    return audio, sample_rate, info

def analyze_audio(ref_file_name: str, synth_file_name: str, file_format: str = "wav"):
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = output_dir / f"{ref_file_name}.{file_format}"
    synth_path = output_dir / f"{synth_file_name}.{file_format}"

    reference, ref_sr, ref_info = load_audio(ref_path)
    synthesized, syn_sr, syn_info = load_audio(synth_path)

    ref_duration = print_audio_info(
        ref_path, reference, ref_sr, ref_info
    )
    syn_duration = print_audio_info(
        synth_path, synthesized, syn_sr, syn_info
    )

    write_file(f"Comparison")
    write_file(f"Duration difference: {syn_duration - ref_duration:+.2f} seconds")
    write_file(f"Reference peak amplitude: {np.max(np.abs(reference)):.4f}")
    write_file(f"Synthesized peak amplitude: {np.max(np.abs(synthesized)):.4f}")

    plot_waveforms(reference, ref_sr, synthesized, syn_sr, output_dir)
    plot_spectra(reference, ref_sr, synthesized, syn_sr, output_dir)
    plot_f0(reference, ref_sr, synthesized, syn_sr, output_dir)

    write_file(f"Plots saved in: {output_dir.resolve()}")