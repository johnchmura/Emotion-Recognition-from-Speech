from pathlib import Path
import pandas as pd

def parse_ravdess_filename(stem: str):
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 fields, got {len(parts)} in '{stem}'")
    return {
        "Modality": parts[0],
        "Vocal Channel": parts[1],
        "Emotion": parts[2],
        "Emotional Intensity": parts[3],
        "Statement": parts[4],
        "Repetition": parts[5],
        "Actor": parts[6],
    }

BASE = Path("data/Audio_Speech_Actors_01-24")
records = []

for wav in BASE.rglob("*.wav"):
    try:
        meta = parse_ravdess_filename(wav.stem)
        meta["File Path"] = str(wav)
        records.append(meta)
    except ValueError as e:
        print(f"Skipping {wav.name}: {e}")

df = pd.DataFrame.from_records(records,
    columns=["File Path","Modality","Vocal Channel","Emotion","Emotional Intensity","Statement","Repetition","Actor"])
df.to_csv("data/cleaned_data.csv", index=False)
print(df)
