Task 3: Work with YourTTS

Objective: to investigate the performance of a pretrained YourTTS model on an unsupported language and prepare a dataset for model adaptation for Lithuanian speech.

1. Preparation for Work

• Python 3.14 can be used with the maintained `coqui-tts` package.

Install the maintained Coqui package and its PyTorch audio codec support:

```powershell
python -m pip install "coqui-tts[codec]"
python -m pip install "transformers==4.57.5"
python -m pip install imageio-ffmpeg pandas
```

`coqui-tts` exposes the same `TTS.api` module used by this project. Pinning
Transformers avoids an incompatibility with newer Transformers releases.
`imageio-ffmpeg` supplies the FFmpeg executable needed for M4A-to-WAV
conversion without requiring a system-wide FFmpeg installation.
Verify the installation with:

```powershell
python --version
python -c "from TTS.api import TTS; print('TTS import successful')"
```

• Load the pretrained YourTTS model:

```python
tts = TTS(
model_name="tts_models/multilingual/multi-dataset/your_tts"
).to(device) Check the languages supported by the pretrained model: print(tts.languages)
```

In the report:

✓ list the supported languages and report whether Lithuanian is available among the supported languages;

2. Zero-Shot Speech Synthesis (English)

Record yourself saying the following sentence: “This is a test of the speech synthesis system.”

Save the recording as “reference_en.wav”. This recording will serve as both a natural speech reference and a reference for the target speaker's voice.

Using the pretrained YourTTS model, synthesize the same sentence:

```python
tts.tts_to_file(
text="This is a test of the speech synthesis system.",
speaker_wav="reference_en.wav",
language="en",
file_path="output.wav"
)
```

In the report:

✓ provide information about the reference recording (duration, sampling frequency, and audio format);

✓ Write your opinion about the quality of the synthesized speech.

✓ plot the waveforms, magnitude spectra, and fundamental frequency (F0) contours of the natural and synthesized signals;

✓ Compare the obtained plots and comment on the observed differences in signal duration, spectral characteristics, and fundamental frequency patterns.

3. Zero-Shot Speech Synthesis (Lithuanian)

Record yourself saying the following Lithuanian sentence: “Tai yra šnekos sintezės sistemos testas.” Save the recording as “reference_lt.wav.”

If you are not a native Lithuanian speaker, you may use another language of your choice that is not supported by the pretrained YourTTS model. In this case, record a short sentence in that language and use the same sentence for speech synthesis.

Using the pretrained YourTTS model, synthesize the same sentence. Since Lithuanian is not supported by the pretrained model, use the English language setting.

In the report:

✓ Write your opinion about the quality of the synthesized speech.

✓ Write your opinion about the quality of the synthesized speech.

✓ plot the waveforms, magnitude spectra, and fundamental frequency (F0) contours of the natural and synthesized signals;

✓ Compare the obtained plots and comment on the observed differences in signal duration, spectral characteristics, and fundamental frequency patterns.

4. Lithuanian Dataset Preparation

Use the Lithuanian Emotional Speech Corpus (EMO).

Emotion labels used in the EMO corpus: D – Joy (Džiaugsmas), L – Sadness (Liūdesys), N – Surprise (Nuostaba), P – Anger (Pyktis), Z – Neutral (Neutrali).

The complete EMO corpus is very large. For this laboratory exercise, do not download the entire corpus. Download only one WAV archive (e.g., emo-wavs.zip.009).

Select:

• one speaker;

• one emotion only;

• corresponding speech recordings and transcriptions.

Split the selected recordings into training (90%) and validation (10%) sets.

Prepare the dataset in the required TTS format:

```text
dataset/
├── wavs/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
├── metadata_train.csv
└── metadata_val.csv
```

The metadata should contain the correspondence between each speech recording and its transcription (read more: https://docs.coqui.ai/en/latest/formatting_your_dataset.html#formatting-your-dataset)

In the report:

✓ specify the selected speaker and emotion;

✓ report the total number and total duration of selected recordings;

✓ provide the main audio parameters (sampling frequency, bit depth, number of channels);

✓ report the minimum, maximum, and average duration of the selected recordings and plot a histogram of recording durations;

✓ report the number of recordings in the training and validation sets.
