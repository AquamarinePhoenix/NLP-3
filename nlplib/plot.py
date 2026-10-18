import json
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch
from nlplib.config import THEME_PATH

def load_powerbi_theme():
    with THEME_PATH.open("r", encoding = "utf-8") as file:
        return json.load(file)

def apply_powerbi_theme():
    theme = load_powerbi_theme()

    colors = theme.get("dataColors", ["#118DFF", "#E66C37"])
    background = theme.get("background", "#FFFFFF")
    foreground = theme.get("foreground", "#252423")
    table_accent = theme.get("tableAccent", "#118DFF")

    text_classes = theme.get("textClasses", {})
    title_style = text_classes.get("title", {})
    label_style = text_classes.get("label", {})

    font = title_style.get("fontFace", "Arial")
    title_size = title_style.get("fontSize", 12)
    label_size = label_style.get("fontSize", 9)
    label_color = label_style.get("color", "#605E5C")

    plt.rcParams.update({
        "figure.facecolor": background,
        "axes.facecolor": background,
        "savefig.facecolor": background,
        "savefig.edgecolor": background,
        "font.family": font,
        "font.size": label_size,
        "text.color": foreground,
        "axes.titlesize": title_size,
        "axes.titleweight": "normal",
        "axes.titlecolor": foreground,
        "axes.labelsize": label_size,
        "axes.labelcolor": label_color,
        "axes.prop_cycle": plt.cycler(color=colors),
        "axes.edgecolor": "#D9D9D9",
        "axes.linewidth": 0.8,
        "xtick.color": label_color,
        "ytick.color": label_color,
        "xtick.labelsize": label_size,
        "ytick.labelsize": label_size,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    return {
        "colors": colors,
        "background": background,
        "foreground": foreground,
        "label_color": label_color,
        "accent": table_accent,
    }

def style_axis(ax, title, xlabel, ylabel, theme):
    ax.set_title(
        title,
        loc="left",
        pad=12,
        color=theme["foreground"],
        fontweight="normal",
    )
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(True, color="#E6E6E6", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9D9D9")
    ax.spines["bottom"].set_color("#D9D9D9")
    ax.tick_params(length=0, pad=6)

def save_plot(fig, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=2.0)
    fig.savefig(
        output_dir / filename,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

def plot_waveforms(reference, ref_sr, synthesized, syn_sr, output_dir):
    theme = apply_powerbi_theme()
    colors = theme["colors"]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=False,
        layout="constrained",
    )

    series = [
        (axes[0], reference, ref_sr, "Natural reference speech", colors[0]),
        (
            axes[1], synthesized, syn_sr,
            "YourTTS synthesized speech", colors[2 % len(colors)]
        ),
    ]

    for ax, audio, sr, title, color in series:
        time = np.arange(len(audio)) / sr

        ax.plot(time, audio, color=color, linewidth=0.7)
        style_axis(ax, title, "Time (seconds)", "Amplitude", theme)

        ax.set_ylim(-1.05, 1.05)

    save_plot(fig, output_dir, "english_waveforms.png")

def plot_spectra(reference, ref_sr, synthesized, syn_sr, output_dir):
    theme = apply_powerbi_theme()
    colors = theme["colors"]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        layout="constrained",
    )

    series = [
        (axes[0], reference, ref_sr, "Natural reference spectrum", colors[0]),
        (
            axes[1], synthesized, syn_sr,
            "YourTTS synthesized spectrum", colors[2 % len(colors)]
        ),
    ]

    for ax, audio, sr, title, color in series:
        if len(audio) < 2:
            continue

        frequencies, power = welch(
            audio,
            fs=sr,
            nperseg=min(4096, len(audio)),
        )
        power_db = 10 * np.log10(power + 1e-12)

        ax.plot(frequencies, power_db, color=color, linewidth=0.9)
        ax.set_xlim(0, sr / 2)

        style_axis(
            ax,
            title,
            "Frequency (Hz)",
            "Power spectral density (dB/Hz)",
            theme,
        )

    save_plot(fig, output_dir, "english_spectra.png")

def plot_f0(reference, ref_sr, synthesized, syn_sr, output_dir):
    theme = apply_powerbi_theme()
    colors = theme["colors"]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        layout="constrained",
    )

    series = [
        (axes[0], reference, ref_sr, "Natural reference F0", colors[0]),
        (
            axes[1], synthesized, syn_sr,
            "YourTTS synthesized F0", colors[2 % len(colors)]
        ),
    ]

    for ax, audio, sr, title, color in series:
        if len(audio) < 2048:
            style_axis(ax, title, "Time (seconds)", "F0 (Hz)", theme)
            ax.text(
                0.5, 0.5,
                "Audio too short for F0 analysis",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=theme["label_color"],
            )
            continue

        f0 = librosa.yin(
            audio,
            fmin=65,
            fmax=400,
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )

        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=sr,
            hop_length=256,
        )

        ax.plot(times, f0, color=color, linewidth=1.1)
        ax.set_ylim(50, 450)

        style_axis(ax, title, "Time (seconds)", "F0 (Hz)", theme)

    save_plot(fig, output_dir, "english_f0.png")