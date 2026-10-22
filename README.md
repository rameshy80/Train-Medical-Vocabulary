# Medical Vocabulary Whisper Fine-Tuning

This project fine-tunes a Whisper model on medical-vocabulary sentences and matching audio recordings.

## Data preparation

1. Generate sentences containing medical vocabulary with a locally run Llama model.
2. Use Google Translate text-to-speech to create audio, selecting the available accent closest to your own.
3. Save each recording and pair it with its matching sentence transcript in a CSV file.

## Dataset format

By default, the script reads `medsentence.csv`. It expects these columns:

| Column | Description |
| --- | --- |
| `transcript` | Text matching the audio recording |
| `audio` or `audio_path` | Path to the recording; `audio_path` is renamed to `audio` |

Audio paths are resolved from the directory where you run the script. Audio is loaded as mono at 16 kHz. The dataset processes recordings one at a time and skips rows it cannot load.

## Configuration

Edit the settings near the top of `train.py`:

- `MODEL_ID`: Pretrained Whisper model or local model directory. The default is `whisper-medical-final`.
- `CSV_PATHS`: List of CSV files to use for training.
- `OUTPUT_DIR`: Directory for training outputs and checkpoints.
- `FINAL_MODEL_DIR`: Directory where the final model and processor are saved.

**Note:** `MODEL_ID` and `FINAL_MODEL_DIR` both default to `whisper-medical-final`. The processor and model must already be loadable from `MODEL_ID` before training starts.

## Run

Install Python packages used by the script: PyTorch, Transformers, pandas, and librosa. Then run:

````powershell
python train.py

The current settings use a batch size of 2, gradient accumulation of 4, a learning rate of 1e-5, and a maximum of 100 training steps. Training uses BF16, so use hardware and a PyTorch setup that support it.

The script saves training outputs under ./whisper-medical-local and the final model and processor under ./whisper-medical-final. It does not configure evaluation.

Limitations
Generated sentences and text-to-speech recordings may contain errors or differ from natural clinical speech. Review the data and model outputs before use. This project is experimental and is not intended for clinical use.
````