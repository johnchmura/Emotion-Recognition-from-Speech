import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from .feature_extraction import augment_audio, extract_features, select_optimal_states
from hmmlearn import hmm
import joblib

def train_emotion_hmms(cfg: dict):
    """
    Train one HMM per emotion class.
    """
    cleaned_csv = Path(cfg["paths"]["cleaned_csv"])
    models_dir  = Path(cfg["paths"]["models_dir"]) / "emotion"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_csv)
    emotions = sorted(df["Emotion"].unique())
    label_map = {emo: idx for idx, emo in enumerate(emotions)}

    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Emotion"],
        random_state=42
    )

    sr        = cfg["mfcc"]["sr"]
    n_mfcc    = cfg["mfcc"]["n_mfcc"]
    frame_len = cfg["mfcc"]["frame_len"]
    hop_len   = cfg["mfcc"]["hop_len"]
    n_mels    = cfg["mfcc"]["n_mels"]

    n_states = cfg["hmm"]["n_states"]
    n_iter   = cfg["hmm"]["n_iter"]


    seqs, lengths, labels = [], [], []
    for _, row in train_df.iterrows():
        wav_path = Path(row["File Path"])
        feats = extract_features(wav_path, sr, n_mfcc, frame_len, hop_len, n_mels)
        seqs.append(feats)
        lengths.append(feats.shape[0])
        labels.append(label_map[row["Emotion"]])

    histories = {}
    for emo in emotions:
        idx    = label_map[emo]
        emo_seqs = [s for s,l in zip(seqs, labels) if l == idx]
        emo_lens = [s.shape[0] for s in emo_seqs]
        X_cat    = np.vstack(emo_seqs)

        print(f"Training emotion HMM '{emo}' on {len(emo_seqs)} sequences...")
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            verbose=False
        )
        model.fit(X_cat, emo_lens)
        histories[emo] = model.monitor_.history

        model_file = models_dir / f"{emo}.pkl"
        joblib.dump(model, model_file)
        print(f"Saved emotion HMM to {model_file}")

    return histories

def train_vocal_hmms(cfg: dict):
    """
    Train one HMM per vocal channel (speech vs. song) using
    leave-actors-out, automatically selecting the number of states
    via BIC.
    """
    cleaned_csv = Path(cfg["paths"]["cleaned_csv"])
    models_dir  = Path(cfg["paths"]["models_dir"]) / "vocal"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_csv)
    channels = sorted(df["Vocal Channel"].unique())
    label_map = {ch: idx for idx, ch in enumerate(channels)}

    # leave-actors-out split
    actors = df["Actor"].unique().tolist()
    train_actors, _ = train_test_split(actors, test_size=0.2, random_state=42)
    train_df = df[df["Actor"].isin(train_actors)].reset_index(drop=True)

    # MFCC params
    sr        = cfg["mfcc"]["sr"]
    n_mfcc    = cfg["mfcc"]["n_mfcc"]
    frame_len = cfg["mfcc"]["frame_len"]
    hop_len   = cfg["mfcc"]["hop_len"]
    n_mels    = cfg["mfcc"]["n_mels"]

    # HMM hyperparams
    state_candidates = cfg["hmm"].get("state_candidates", [2,3,5,7,10])
    n_iter    = cfg["hmm"]["n_iter"]

    seqs, lengths, labels = [], [], []
    for _, row in train_df.iterrows():
        wav_path = Path(row["File Path"])
        feats = extract_features(wav_path, sr, n_mfcc, frame_len, hop_len, n_mels)
        seqs.append(feats)
        lengths.append(feats.shape[0])
        labels.append(label_map[row["Vocal Channel"]])

    histories = {}
    for ch in channels:
        idx      = label_map[ch]
        ch_seqs  = [s for s,l in zip(seqs, labels) if l == idx]
        ch_lens  = [s.shape[0] for s in ch_seqs]
        X_cat    = np.vstack(ch_seqs)
        D        = X_cat.shape[1]

        print(f"\nSelecting n_states for '{ch}' from {state_candidates} …")
        sel = select_optimal_states(
            X_cat, ch_lens, D,
            state_list=state_candidates,
            n_iter=n_iter
        )
        best_k = sel["bic"]
        print(f"→ BIC-optimal states for '{ch}': {best_k}")

        print(f"Training vocal-channel HMM '{ch}' with {best_k} states …")
        model = hmm.GaussianHMM(
            n_components=best_k,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-4,
            verbose=False
        )
        model.fit(X_cat, ch_lens)
        histories[ch] = model.monitor_.history

        model_file = models_dir / f"{ch}.pkl"
        joblib.dump(model, model_file)
        print(f"Saved HMM for '{ch}' to {model_file}")

    return histories
