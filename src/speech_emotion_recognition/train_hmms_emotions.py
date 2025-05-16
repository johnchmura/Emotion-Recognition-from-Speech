import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import soundfile as sf
import librosa
from hmmlearn import hmm

SR        = 16000
N_MFCC    = 13
FRAME_LEN = 0.025
HOP_LEN   = 0.010
N_MELS    = 40
N_STATES  = 5
N_ITER    = 30

CSV_PATH  = Path("data/cleaned_data.csv")

df = pd.read_csv(CSV_PATH)
classes = sorted(df["Emotion"].unique())
label_map = {emo: i for i, emo in enumerate(classes)}
print("Emotions:", classes)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Emotion"],
    random_state=42
)

def extract_features_from_file(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=int(sr*FRAME_LEN),
        hop_length=int(sr*HOP_LEN),
        n_mels=N_MELS, fmax=sr/2, htk=True
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2]).T
    mu = feats.mean(0, keepdims=True)
    sigma = feats.std(0, keepdims=True) + 1e-9
    return (feats - mu) / sigma

train_seqs, train_lens, train_labels = [], [], []
print("Preparing training data...")
for _, row in train_df.iterrows():
    feats = extract_features_from_file(row["File Path"])
    train_seqs.append(feats)
    train_lens.append(feats.shape[0])
    train_labels.append(label_map[row["Emotion"]])

models, histories = {}, {}
for emo, cls in enumerate(classes):
    seqs = [s for s, l in zip(train_seqs, train_labels) if l == emo]
    lengths = [s.shape[0] for s in seqs]
    X_cat = np.vstack(seqs)
    print(f"Training HMM for '{cls}' on {len(seqs)} examples…")
    m = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        verbose=False
    )
    m.fit(X_cat, lengths)
    models[emo]    = m
    histories[emo] = m.monitor_.history

print("Scoring test set…")
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    feats = extract_features_from_file(row["File Path"])
    scores = [models[c].score(feats) for c in range(len(classes))]
    pred   = int(np.argmax(scores))
    y_pred.append(pred)
    y_true.append(label_map[row["Emotion"]])
y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot()
plt.title("HMM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

plt.figure()
for emo, cls in enumerate(classes):
    plt.plot(histories[emo], label=cls)
plt.title("HMM EM Log-Likelihood per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.legend()
plt.tight_layout()
plt.show()

accuracy = cm.trace() / cm.sum()
print(f"Overall Accuracy: {accuracy*100:.2f}%")
