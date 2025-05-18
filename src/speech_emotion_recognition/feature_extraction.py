import random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import numpy as np
from hmmlearn import hmm

def select_optimal_states(X_cat, lengths, D, state_list, n_iter=30):
    """
    Grid‑search over `state_list` of hidden‑state counts.
    Returns dict with best_n_states by AIC and BIC, plus full table.

    Args:
      X_cat   : np.ndarray, shape (N_frames, D) — all frames stacked
      lengths : list[int]      — sequence lengths per utterance
      D       : int            — feature dimension
      state_list : list[int]   — e.g. [2,3,5,7,10]
      n_iter  : int            — EM max iterations

    Returns:
      {
        "aic":    best_n_states_for_AIC,
        "bic":    best_n_states_for_BIC,
        "results": [
           { "n_states": k,
             "logL":    ℓ,
             "p":       num_params,
             "AIC":     aic,
             "BIC":     bic
           }, …
        ]
      }
    """
    N = X_cat.shape[0]
    results = []

    for k in state_list:
        # fit HMM
        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-4,
            verbose=False
        )
        model.fit(X_cat, lengths)
        logL = model.score(X_cat, lengths)

        # count parameters:
        #   initial probs: (k−1)
        #   transitions: k*(k−1)
        #   emissions: means k*D + variances k*D
        p = (k - 1) + k*(k - 1) + 2*k*D

        # compute AIC & BIC
        aic = -2 * logL + 2 * p
        bic = -2 * logL + p * np.log(N)

        results.append({
            "n_states": k,
            "logL":     logL,
            "p":        p,
            "AIC":      aic,
            "BIC":      bic
        })
        print(f"States={k}: logL={logL:.1f}, p={p}, AIC={aic:.1f}, BIC={bic:.1f}")

    # pick minima
    best_aic = min(results, key=lambda r: r["AIC"])["n_states"]
    best_bic = min(results, key=lambda r: r["BIC"])["n_states"]

    return {"aic": best_aic, "bic": best_bic, "results": results}

def preprocess_audio(path: str,target_sr: int = 16_000,eps: float = 1e-9,pre_emphasis: float = 0.97) -> np.ndarray:
    """
    Load, mono‑mix, DC‑shift, pre‑emphasize, resample, and normalize a WAV.
    Returns a 1D float array in [-1,1].
    """
    audio, orig_sr = sf.read(path)
    # mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # zero‑mean
    audio = audio - np.mean(audio)
    # pre‑emphasis
    audio = np.concatenate(([audio[0]], audio[1:] - pre_emphasis * audio[:-1]))
    # resample
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    # peak normalize
    peak = np.max(np.abs(audio))
    return audio / (peak + eps)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to a signal and re‑normalize."""
    noise = np.random.randn(len(audio))
    aug   = audio + noise_factor * noise
    return aug / (np.max(np.abs(aug)) + 1e-9)

def volume_perturb(audio: np.ndarray, vol_range: tuple[float,float]) -> np.ndarray:
    factor = random.uniform(*vol_range)
    aug    = audio * factor
    return aug / (np.max(np.abs(aug)) + 1e-9)

def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Speed up or slow down without changing pitch."""
    return librosa.effects.time_stretch(audio, rate = rate)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_shift(audio: np.ndarray, max_shift_sec: float, sr: int) -> np.ndarray:
    max_shift = int(max_shift_sec * sr)
    shift     = random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)

def augment_audio(audio: np.ndarray,sr: int,noise_prob: float= 0.9,noise_range: tuple[float,float]= (0.01, 0.05),stretch_prob: float= 0.8,
    stretch_range: tuple[float,float] = (0.8, 1.2),pitch_prob: float = 0.8,pitch_range: tuple[float,float] = (-5, 5),volume_prob: float     = 0.7,
    volume_range: tuple[float,float] = (0.7, 1.3),shift_prob: float= 0.5,shift_max_sec: float = 0.2,reverse_prob: float    = 0.3
) -> np.ndarray:
    """
      - Gaussian noise
      - Time-stretch
      - Pitch-shift
      - Volume scaling
      - Time shift
      - Random reversal
    """
    aug = audio.copy()

    if random.random() < reverse_prob:
        aug = aug[::-1]

    if random.random() < noise_prob:
        nf = random.uniform(*noise_range)
        aug = add_noise(aug, nf)

    if random.random() < stretch_prob:
        rate = random.uniform(*stretch_range)
        aug  = time_stretch(aug, rate)

    if random.random() < pitch_prob:
        steps = random.uniform(*pitch_range)
        aug   = pitch_shift(aug, sr, steps)

    if random.random() < volume_prob:
        aug   = volume_perturb(aug, volume_range)

    if random.random() < shift_prob:
        aug   = time_shift(aug, shift_max_sec, sr)

    return aug



def extract_features(wav_path: Path,sr: int = 16_000,n_mfcc: int = 13,frame_len: float = 0.025,hop_len: float = 0.010,n_mels: int = 40,
    cmvn_eps: float = 1e-9) -> np.ndarray:
    """
    Load a WAV and compute MFCC + delta + delta‑delta, then apply
    cepstral mean‑variance normalization (per utterance).
    Returns a (n_frames, 3*n_mfcc) array.
    """
    audio, sample_rate = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(sample_rate * frame_len),
        hop_length=int(sample_rate * hop_len),
        n_mels=n_mels,
        fmax=sample_rate / 2,
        htk=True
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2]).T

    # CMVN
    mean = feats.mean(axis=0, keepdims=True)
    std  = feats.std(axis=0, keepdims=True) + cmvn_eps
    return (feats - mean) / std


def preprocess_all(raw_base: Path, preproc_base: Path, sr: int = 16_000) -> None:
    """
    Walk `raw_base`, apply `preprocess_audio` to each WAV, and write
    the result under `preproc_base`, preserving subdirectories.
    """
    preproc_base.mkdir(parents=True, exist_ok=True)
    for wav in raw_base.rglob("*.wav"):
        out_path = preproc_base / wav.relative_to(raw_base)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = preprocess_audio(str(wav), target_sr=sr)
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
    print(f"Preprocessed audio now in {preproc_base}")
