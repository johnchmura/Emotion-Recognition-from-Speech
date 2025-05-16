import soundfile as sf
import librosa
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_audio(path, target_sr=16000, top_db=40, eps=1e-9, pre_emph=0.97):
    """
    Load audio from path, convert to mono, remove DC offset, resample
    to target_sr, trim leading/trailing silence, and perform max-abs normalization.
    Returns:
      audio: 1D numpy array, peak-normalized in [-1, 1]
      sr: sampling rate after resampling (target_sr)
    """
    audio, orig_sr = sf.read(path)
    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Remove DC offset
    audio = audio - np.mean(audio)

    audio = np.concatenate(([audio[0]], audio[1:] - pre_emph * audio[:-1]))

    # Resample to target_sr
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    # Peak-normalize
    peak = np.max(np.abs(audio))
    audio = audio / (peak + eps)
    #audio, _ = librosa.effects.trim(audio, top_db=top_db)
    max_len = int(2.4 * target_sr)
    #audio_fixed = pad_or_truncate(audio, max_len)
    return audio, target_sr


def add_noise(audio, noise_factor=0.005):
    """Add Gaussian noise to the signal."""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented / np.max(np.abs(augmented))

def stretch(audio, rate):
    """Time-stretch the audio by a factor `rate`."""
    return librosa.effects.time_stretch(audio, rate=rate)

def shift(audio, sr, n_steps):
    """Pitch-shift the audio by `n_steps` (in semitones)."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def pad_or_truncate(audio, max_len):
    """Pad with zeros or truncate to ensure audio length == max_len."""
    if len(audio) > max_len:
        return audio[:max_len]
    else:
        return np.pad(audio, (0, max_len - len(audio)), mode='constant')

def augment_audio(audio, sr,noise_prob=0.5, noise_factor=0.005,stretch_prob=0.5, stretch_range=(0.9, 1.1),pitch_prob=0.5, pitch_steps=(-2, 2)):
    """
    Randomly apply augmentations:
      - Gaussian noise
      - Time-stretch
      - Pitch-shift
    Each with its own probability.
    """
    aug = audio.copy()
    if random.random() < noise_prob:
        aug = add_noise(aug, noise_factor)
    if random.random() < stretch_prob:
        rate = random.uniform(*stretch_range)
        aug = stretch(aug, rate)
    if random.random() < pitch_prob:
        steps = random.uniform(*pitch_steps)
        aug = shift(aug, sr, steps)
    return aug


def extract_mfcc(path, sr=16000, n_mfcc=13, frame_len=0.025, hop_len=0.010, cmvn_eps=1e-9):
    """
    Extract MFCC + delta + delta-delta from a WAV file and apply CMVN.
    Returns:
      feats: np.ndarray of shape (n_frames, 3 * n_mfcc), zero-mean & unit-variance per feature.
    """
    audio, _ = sf.read(path)
    
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=int(sr * frame_len),
        hop_length=int(sr * hop_len),
        htk=True
    )
    
    # Compute first and second derivatives
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    coeffs = np.vstack([mfcc, delta, delta2])
    
    feats = coeffs.T
    
    mean = np.mean(feats, axis=0, keepdims=True)
    std  = np.std(feats,  axis=0, keepdims=True)
    feats_cmvn = (feats - mean) / (std + cmvn_eps)
    
    return feats_cmvn

def extract_feats(wav_path: Path, sr: int, n_mfcc: int,
                   frame_len: float, hop_len: float,
                   n_mels: int) -> np.ndarray:
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
        fmax=sample_rate/2,
        htk=True
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2]).T
    mu    = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-9
    return (feats - mu) / sigma

def preprocess_all(raw_base: Path, preproc_base: Path, sr: int):
    preproc_base.mkdir(parents=True, exist_ok=True)
    for wav in Path(raw_base).rglob("*.wav"):
        audio, _ = preprocess_audio(str(wav), target_sr=sr)
        out_path = preproc_base / wav.relative_to(raw_base)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"Preprocessed audio written under {preproc_base}")