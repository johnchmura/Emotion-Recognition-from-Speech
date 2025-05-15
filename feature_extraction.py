import soundfile as sf
import librosa
import numpy as np
import random
import numpy as np

def preprocess_audio(path, target_sr=16000, top_db=10, eps=1e-9, pre_emph=0.97):
    """
    Load audio from path, convert to mono, remove DC offset, resample
    to target_sr, trim leading/trailing silence, and perform max-abs normalization.
    Returns:
      audio: 1D numpy array, peak-normalized in [-1, 1]
      sr: sampling rate after resampling (target_sr)
    """
    audio, orig_sr = sf.read(path)
    #Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    #Remove DC offset
    audio = audio - np.mean(audio)

    audio = np.concatenate(([audio[0]], audio[1:] - pre_emph * audio[:-1]))

    #Resample to target_sr
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    #Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    #Peak-normalize
    peak = np.max(np.abs(audio))
    audio = audio / (peak + eps)
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

def preprocess_and_augment(path, target_sr=16000, **kwargs):
    audio, sr = preprocess_audio(path, target_sr=target_sr)
    audio_aug = augment_audio(audio, sr, **kwargs)
    max_len = int(1.5 * sr)
    audio_fixed = pad_or_truncate(audio_aug, max_len)
    return audio_fixed

demo_path = "data/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
import matplotlib.pyplot as plt

augmented_audio = preprocess_and_augment(demo_path, noise_factor=0.01, stretch_range=(0.95, 1.05))

plt.figure(figsize=(10, 4))
plt.plot(augmented_audio)
plt.title("Waveform of Augmented Audio")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
print("Processed length:", len(augmented_audio))

