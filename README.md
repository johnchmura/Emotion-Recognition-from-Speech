Thought for a couple of seconds

<!-- README.md -->

# Speech Emotion & Vocal‑Channel Recognition

A Python toolkit for training and evaluating **Hidden Markov Models** (HMMs) on speech data to perform:

- **Emotion Recognition** (neutral, calm, happy, sad, angry, fearful, disgust, surprised)  
- **Vocal‑Channel Classification** (speech vs. song)

This package provides end‑to‑end data parsing, preprocessing, feature extraction, model training, and evaluation via a simple Command‑Line Interface (`ser`).

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your‑org/speech_emotion_recognition.git
   cd speech_emotion_recognition
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode**  
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

---

## ⚙️ Configuration

All paths and hyperparameters live in `config.toml` at the project root:

```toml
[paths]
raw_data_dir     = "data/Audio_Speech_Actors_01-24"
preproc_dir      = "data/preprocessed"
cleaned_csv      = "data/cleaned_data.csv"
models_dir       = "models"

[mfcc]
sr        = 16000
n_mfcc    = 13
frame_len = 0.025
hop_len   = 0.010
n_mels    = 40

[hmm]
n_states = 5
n_iter   = 30
```

Feel free to tweak sampling rate, MFCC settings, or HMM states and iterations.

---

## 🚀 Usage

After installation, you have a single CLI entrypoint:  

```bash
$ ser --help
```

### 1. Preprocess

Parse raw RAVDESS filenames into a metadata CSV and resample + normalize all WAVs:

```bash
ser preprocess
```

- **Output**:  
  - `data/cleaned_data.csv` (one row per utterance with parsed fields)  
  - `data/preprocessed/...` (mono 16 kHz, pre‑emphasized, normalized WAVs)

### 2. Train

Train HMMs in one of two modes:

```bash
# Emotion models (one HMM per emotion)
ser train --mode emotion

# Vocal‑channel models (one HMM per channel: speech vs. song)
ser train --mode vocal
```

Trained model files will be saved under:

```
models/
├── emotion/
│   ├── angry.pkl
│   ├── calm.pkl
│   └── … 
└── vocal/
    ├── speech.pkl
    └── song.pkl
```

### 3. Evaluate

Evaluate held‑out performance and plot a confusion matrix:

```bash
# Emotion recognition evaluation
ser eval --mode emotion

# Vocal-channel evaluation
ser eval --mode vocal
```

---

## 🔍 Model Methodology

1. **Filename Parsing**  
   - RAVDESS filenames (e.g. `03-01-06-02-02-02-15.wav`) are split into fields:  
     modality, vocal channel, emotion, intensity, statement, repetition, actor  
   - Stored in `data/cleaned_data.csv`.

2. **Preprocessing**  
   - **Mono conversion**, **DC‑offset removal**, **pre‑emphasis** (α=0.97)  
   - **Resampling** to 16 kHz, **peak normalization**  
   - Silence trimming is optional (controlled in code).

3. **Feature Extraction**  
   - **MFCCs** (13 coefficients) computed on 25 ms frames with 10 ms hop  
   - **Deltas** & **delta‑deltas** stacked → 3×13 features per frame  
   - **Cepstral Mean–Variance Normalization** per utterance

4. **Hidden Markov Models**  
   - One **GaussianHMM** (diagonal covariance) per class  
   - Trained with **EM** for a fixed number of states (configurable)  
   - **Emotion HMMs** use a stratified train/test split by emotion  
   - **Vocal HMMs** use a leave‑actors‑out scheme for robust generalization  

5. **Evaluation**  
   - Log‑likelihood scoring on test sequences  
   - **Confusion matrix** and **overall accuracy** metrics  

---

## 📂 Project Structure

```
speech_emotion_recognition/
├── setup.py                 # setuptools install script
├── config.toml              # project & model configuration
├── src/
│   └── speech_emotion_recognition/
│       ├── __init__.py
│       ├── ser.py           # CLI entrypoint
│       ├── data_parser.py   # filename → metadata
│       ├── feature_extraction.py  # audio preprocess
│       ├── train.py         # train_emotion_hmms & train_vocal_hmms
│       └── evaluate.py      # evaluate_emotion_hmms & evaluate_vocal_hmms
└── data/                    # raw & preprocessed data (not committed)
```

---

## 🧪 Testing

Use pytest for unit tests:

```bash
pip install pytest
pytest
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.