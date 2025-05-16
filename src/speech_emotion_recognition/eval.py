import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from .feature_extraction import extract_features
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib

def evaluate_emotion_hmms(cfg: dict):
    """
    Load & evaluate one HMM per emotion; stratified split by emotion.
    """
    cleaned_csv = Path(cfg["paths"]["cleaned_csv"])
    models_dir  = Path(cfg["paths"]["models_dir"]) / "emotion"

    df = pd.read_csv(cleaned_csv)
    emotions = sorted(df["Emotion"].unique())
    label_map = {emo: idx for idx, emo in enumerate(emotions)}

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Emotion"],
        random_state=42
    )

    models = { emo: joblib.load(models_dir / f"{emo}.pkl") for emo in emotions }

    sr        = cfg["mfcc"]["sr"]
    n_mfcc    = cfg["mfcc"]["n_mfcc"]
    frame_len = cfg["mfcc"]["frame_len"]
    hop_len   = cfg["mfcc"]["hop_len"]
    n_mels    = cfg["mfcc"]["n_mels"]

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        wav_path = Path(row["File Path"])
        feats    = extract_features(wav_path, sr, n_mfcc, frame_len, hop_len, n_mels)
        scores   = [models[e].score(feats) for e in emotions]
        pred     = int(np.argmax(scores))
        y_true.append(label_map[row["Emotion"]])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(emotions)))
    disp = ConfusionMatrixDisplay(cm, display_labels=emotions)
    disp.plot(cmap="Blues")
    plt.title("Emotion HMM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_true, y_pred)
    print(f"Emotion HMM Accuracy: {acc*100:.2f}%\n")

def evaluate_vocal_hmms(cfg: dict):
    """
    Load & evaluate one HMM per vocal channel (speech/song); leave-actors-out split.
    """
    cleaned_csv = Path(cfg["paths"]["cleaned_csv"])
    models_dir  = Path(cfg["paths"]["models_dir"]) / "vocal"

    df = pd.read_csv(cleaned_csv)
    channels = sorted(df["Vocal Channel"].unique())
    label_map = {ch: idx for idx, ch in enumerate(channels)}

    actors = df["Actor"].unique().tolist()
    _, test_actors = train_test_split(
        actors,
        test_size=0.2,
        random_state=42
    )
    test_df = df[df["Actor"].isin(test_actors)].reset_index(drop=True)

    models = { ch: joblib.load(models_dir / f"{ch}.pkl") for ch in channels }

    sr        = cfg["mfcc"]["sr"]
    n_mfcc    = cfg["mfcc"]["n_mfcc"]
    frame_len = cfg["mfcc"]["frame_len"]
    hop_len   = cfg["mfcc"]["hop_len"]
    n_mels    = cfg["mfcc"]["n_mels"]

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        wav_path = Path(row["File Path"])
        feats    = extract_features(wav_path, sr, n_mfcc, frame_len, hop_len, n_mels)
        scores   = [models[c].score(feats) for c in channels]
        pred     = int(np.argmax(scores))
        y_true.append(label_map[row["Vocal Channel"]])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(channels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=channels)
    disp.plot(cmap="Greens")
    plt.title("Vocal Channel HMM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_true, y_pred)
    print(f"Vocal Channel HMM Accuracy: {acc*100:.2f}%\n")
