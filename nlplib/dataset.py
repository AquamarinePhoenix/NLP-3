import random
import os
import shutil
import stat
import soundfile as sf
import matplotlib.pyplot as plt
import struct
import pandas as pd
import zlib
from nlplib.config import INPUT_DIR, SPEAKER_ID, EMOTION, RANDOM_SEED, DATASET_DIR
from nlplib.write import write_file

def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def zip_to_df(path):
    size = path.stat().st_size
    entries = []
    with open(path, "rb") as f:
        o = f.read(4 << 20).find(b"PK\x03\x04")
        while o < size - 30:
            f.seek(o); h = f.read(30)
            if h[:4] != b"PK\x03\x04":
                break
            _, flag, meth, _, _, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", h[4:30])
            name = f.read(nlen).decode("utf8", "replace")
            data_off = o + 30 + nlen + elen
            entries.append(dict(name=name, offset=data_off, csize=csize, usize=usize, method=meth, crc=crc,
                                complete=data_off + csize <= size))
            o = data_off + csize + (16 if flag & 8 else 0)
    return pd.DataFrame(entries)

def access_dataset():
    file_loc = INPUT_DIR / "emo-wavs.zip-001.009"
    metadata_loc = INPUT_DIR / "metadata_of_EMO_dataset.csv"

    idx = zip_to_df(file_loc)
    idx = idx[idx["name"].str.endswith(".wav") & idx["complete"]].copy()

    idx["emotion"] = idx["name"].str.split("/").str[1]
    idx["session"] = idx["name"].str.split("/").str[3]
    idx["speaker"] = idx["session"].str[:2]

    meta = pd.read_csv(
        metadata_loc,
        sep="|",
        header=None,
        names=["path", "speaker", "text"],
        dtype=str,
        quoting=3,
    )
    meta["emotion"] = meta["path"].str.split("/").str[1]

    archive_counts = (
        idx.groupby(["emotion", "speaker"])
        .size()
        .rename("in_archive")
        .reset_index()
    )

    metadata_counts = (
        meta.groupby(["emotion", "speaker"])
        .size()
        .rename("in_corpus")
        .reset_index()
    )

    coverage = archive_counts.merge(
        metadata_counts,
        on=["emotion", "speaker"],
        how="left",
    )

    coverage["complete"] = (
        coverage["in_archive"] == coverage["in_corpus"]
    )

    return idx, meta, coverage

def prepare_dataset():
    archive_path = INPUT_DIR / "emo-wavs.zip-001.009"

    idx, meta, _ = access_dataset()

    selected_meta = meta[
        (meta["speaker"] == SPEAKER_ID)
        & (meta["emotion"] == EMOTION)
    ].copy()

    selected_idx = idx[
        (idx["speaker"] == SPEAKER_ID)
        & (idx["emotion"] == EMOTION)
    ].copy()

    selected = selected_meta.merge(
        selected_idx,
        left_on="path",
        right_on="name",
        how="inner",
    )

    if selected.empty:
        raise RuntimeError(
            f"No matching WAVs for speaker {SPEAKER_ID}, "
            f"emotion {EMOTION}."
        )

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR, onerror=_remove_readonly)

    wav_dir = DATASET_DIR / "wavs"
    wav_dir.mkdir(parents=True)

    durations = []
    metadata_rows = []

    
    for number, row in enumerate(
        selected.sample(frac=1, random_state=RANDOM_SEED).to_dict("records"),
        start=1,
    ):
        output_name = f"{number:04d}.wav"
        output_path = wav_dir / output_name

        try:
            extract_wav_from_part(archive_path, row, output_path)

            info = sf.info(str(output_path))
            duration = info.duration

        except Exception as exc:
            output_path.unlink(missing_ok=True)
            write_file(f"Skipping {row['name']}: {exc}")
            continue

        durations.append(duration)
        metadata_rows.append({
            "filename": f"wavs/{output_name}",
            "text": row["text"],
            "duration": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
        })


    if len(metadata_rows) < 2:
        raise RuntimeError("Fewer than two valid audio files were extracted.")

    random.Random(RANDOM_SEED).shuffle(metadata_rows)
    split_at = int(len(metadata_rows) * 0.9)

    train_rows = metadata_rows[:split_at]
    val_rows = metadata_rows[split_at:]

    for filename, rows in [
        ("metadata_train.csv", train_rows),
        ("metadata_val.csv", val_rows),
    ]:
        with (DATASET_DIR / filename).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            for row in rows:
                text = row["text"].replace("\n", " ").replace("|", " ")
                file.write(f"{row['filename']}|{text}\n")

    stats = pd.DataFrame(metadata_rows)
    total_duration = stats["duration"].sum()

    write_file(f"Dataset preparation summary")
    write_file(f"Speaker: {SPEAKER_ID}")
    write_file(f"Emotion: {EMOTION} (Surprise / Nuostaba)")
    write_file(f"Valid recordings: {len(stats)}")
    write_file(f"Total duration: {total_duration:.2f} seconds")
    write_file(f"Minimum duration: {stats['duration'].min():.2f} seconds")
    write_file(f"Maximum duration: {stats['duration'].max():.2f} seconds")
    write_file(f"Average duration: {stats['duration'].mean():.2f} seconds")
    write_file(f"Training recordings: {len(train_rows)}")
    write_file(f"Validation recordings: {len(val_rows)}")

    write_file(f"Audio parameters:")
    write_file(f"Sample rates: {stats['sample_rate'].value_counts().to_dict()}")
    write_file(f"Channels: {stats['channels'].value_counts().to_dict()}")
    write_file(f"Subtypes: {stats['subtype'].value_counts().to_dict()}")

    plt.figure(figsize=(9, 5))
    plt.hist(stats["duration"], bins=30, edgecolor="black")
    plt.title(f"Recording durations — speaker {SPEAKER_ID}, emotion {EMOTION}")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of recordings")
    plt.tight_layout()
    plt.savefig(DATASET_DIR / "duration_histogram.png", dpi=150)
    plt.close()

    write_file(f"Histogram saved to: {DATASET_DIR / 'duration_histogram.png'}")

def extract_wav_from_part(archive_path, row, output_path):
    with open(archive_path, "rb") as archive:
        archive.seek(int(row["offset"]))
        compressed = archive.read(int(row["csize"]))

    method = int(row["method"])

    if method == 0:
        audio_bytes = compressed
    elif method == 8:
        audio_bytes = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported ZIP compression method: {method}")

    if len(audio_bytes) != int(row["usize"]):
        raise ValueError(
            f"Unexpected decompressed size for {row['name']}: "
            f"expected {row['usize']}, got {len(audio_bytes)}"
        )

    crc = zlib.crc32(audio_bytes) & 0xFFFFFFFF
    if crc != int(row["crc"]):
        raise ValueError(f"CRC check failed for {row['name']}")

    with open(output_path, "wb") as output:
        output.write(audio_bytes)