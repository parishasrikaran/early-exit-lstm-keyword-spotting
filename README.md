# Early-Exit LSTM Keyword Spotting — Version 3

This project is written in Python and can be opened in VS Code. Two LSTM models **have been trained** on locally generated synthetic speech (eSpeak) to test the full training-to-evaluation pipeline. It is **not yet a trained real-human-speech recognition system**.

## What is included

- `models/synthetic_baseline.pt`: trained standard LSTM checkpoint.
- `models/synthetic_prefix.pt`: trained LSTM with additional 600 ms and 800 ms training targets.
- `data/synthetic_tts.pt`: the small synthetic training/validation/test dataset, with separate text-to-speech voices in each split.
- `results/synthetic_comparison.json`: measured held-out synthetic test results.
- `results/policy_synthetic_*.json`: validation-selected early-exit thresholds.
- `results/*.png`: validation accuracy, stopping-policy trade-off, and test confusion-matrix plots.
- `examples/tts_*.wav`: several test WAV files.
- `mic_demo.py`: one-utterance microphone demonstration (not an always-listening application).
- `dataset.py`, `train.py`, `tune.py`, `evaluate.py`: scripts for real Google Speech Commands data, if you can download it on your Mac.

## Set up on a Mac

Open this folder in VS Code. Open a terminal in the same folder and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

If Python 3.11 is not installed, install it and reopen the terminal. On some Macs, `sounddevice` needs PortAudio: `brew install portaudio`.

## Verify the results that are already saved

The checkpoints and synthetic dataset are **already in the ZIP**. No retraining or dataset download is needed to run these commands:

```bash
python evaluate.py --dataset synthetic --checkpoints models/synthetic_baseline.pt models/synthetic_prefix.pt --output results/synthetic_comparison.json --device cpu
python streaming.py examples/tts_stop_en-sc.wav --checkpoint models/synthetic_prefix.pt --threshold 0.99
```

The second command may say **No confident keyword**, which is expected at the strict threshold; this does not indicate a broken program.

To generate the plots again, run:

```bash
python analyze_results.py
```

## Try your Mac microphone (experimental)

```bash
python mic_demo.py --checkpoint models/synthetic_prefix.pt --policy results/policy_synthetic_prefix.json
```

Press Enter, speak one command, and allow microphone access if macOS asks. The classifier was trained on synthetic TTS voices, so predictions on your voice may be incorrect, or it may refuse to predict. This demo records **one audio event** and is not a continuously running wake-word listener.

## Retrain the synthetic example (optional)

Generating speech data requires `espeak` (`brew install espeak` on macOS). The saved synthetic dataset is already included, so you can retrain directly:

```bash
python train.py --dataset synthetic --mode baseline --epochs 26 --device cpu --output models/my_baseline.pt
python train.py --dataset synthetic --mode prefix --epochs 26 --device cpu --output models/my_prefix.pt
python tune.py --dataset synthetic --checkpoint models/my_baseline.pt
python tune.py --dataset synthetic --checkpoint models/my_prefix.pt
```

Results may change if your system, library versions, random seed, or hardware differ.

## Next: real human-speech dataset

For meaningful research, first train on Google Speech Commands v2 (download requires internet), then tune thresholds on validation audio and test on held-out human speakers:

```bash
python train.py --dataset speech_commands --mode baseline --epochs 15
python train.py --dataset speech_commands --mode prefix --epochs 15
python tune.py --dataset speech_commands --checkpoint models/baseline.pt
python tune.py --dataset speech_commands --checkpoint models/prefix.pt
python evaluate.py --dataset speech_commands
```

The initial real-data loader is intended as a starting point; verify its dataset split and augmentation behavior before treating its results as publishable.

## Measured results so far (synthetic TTS only)

Both models were trained for up to 26 epochs, with the best validation checkpoint retained. The experiment uses 864 train clips, 288 validation clips, and 288 test clips, across 12 classes.

| Measure | Standard LSTM | Prefix LSTM |
|---|---:|---:|
| Full-window test accuracy | **70.1%** | **76.4%** |
| Early correct keyword recall (240 positive test clips) | **4.6%** | **5.0%** |
| Wrong early keyword rate (positive test clips) | **0.0%** | **2.1%** |
| Early trigger rate (48 negative test clips) | **2.1%** | **0.0%** |

The thresholds were chosen using only the validation set. Emission timestamps in the result files are relative to the start of synthetic one-second clips, **not annotated speech-onset delay**. These figures are not real-human accuracy, real-stream false accepts per hour, energy usage, or evidence of a useful speedup. Early correct recall is currently too low to claim success.

## Main limitations and next tasks

1. Train with real recorded human speech and appropriate train/validation/test separation.
2. Add more realistic partial-word negative samples and optimize the stopping rule to avoid missing nearly every keyword.
3. Build continuous-audio replay with event boundaries; report false accepts/hour and true onset/offset-based delay.
4. Run noise robustness studies, multiple random seeds, and on-device timing measurements.
5. Later, explore a learned stop/defer head and optional iPhone deployment.

For exact metrics, use `results/synthetic_comparison.json`. For a human-readable interpretation, see `RESULTS.md`.
