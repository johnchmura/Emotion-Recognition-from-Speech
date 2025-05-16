from pathlib import Path
import pandas as pd

def parse_ravdess_filename(stem: str):
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 fields, got {len(parts)} in '{stem}'")
    
    match parts[0]:
        case "01":
            parts[0] = "full-AV"
        case "02":
            parts[0] = "video-only"
        case "03":
            parts[0] = "audio-only"
        case _:
            raise ValueError(f"Unknown modality '{parts[0]}' in '{stem}'")
        
    match parts[1]:
        case "01":
            parts[1] = "speech"
        case "02":
            parts[1] = "song"
        case _:
            raise ValueError(f"Unknown vocal channel '{parts[1]}' in '{stem}'")
    
    match parts[2]:
        case "01":
            parts[2] = "neutral"
        case "02":
            parts[2] = "calm"
        case "03":
            parts[2] = "happy"
        case "04":
            parts[2] = "sad"
        case "05":
            parts[2] = "angry"
        case "06":
            parts[2] = "fearful"
        case "07":
            parts[2] = "disgust"
        case "08":
            parts[2] = "surprised"
        case _:
            raise ValueError(f"Unknown emotion '{parts[2]}' in '{stem}'")
    
    match parts[3]:
        case "01":
            parts[3] = "normal"
        case "02":
            parts[3] = "strong"
        case _:
            raise ValueError(f"Unknown emotional intensity '{parts[3]}' in '{stem}'")
    match parts[4]:
        case "01":
            parts[4] = "Kids are talking by the door"
        case "02":
            parts[4] = "Dogs are sitting by the door"
        case _:
            raise ValueError(f"Unknown statement type '{parts[4]}' in '{stem}'")
        
    match parts[5]:
        case "01":
            parts[5] = "1st repetition"
        case "02":
            parts[5] = "2nd repetition"
        case _:
            raise ValueError(f"Unknown repetition '{parts[5]}' in '{stem}'")
    
    return {
        "Modality": parts[0],
        "Vocal Channel": parts[1],
        "Emotion": parts[2],
        "Emotional Intensity": parts[3],
        "Statement": parts[4],
        "Repetition": parts[5],
        "Actor": parts[6],
    }

def parse_all(raw_base: Path, cleaned_csv: Path):
    records = []
    for wav in Path(raw_base).rglob("*.wav"):
        meta = parse_ravdess_filename(wav.stem)
        meta["File Path"] = str(wav)
        records.append(meta)
    df = pd.DataFrame.from_records(records, columns=[
        "File Path","Modality","Vocal Channel","Emotion",
        "Emotional Intensity","Statement","Repetition","Actor"
    ])
    df.to_csv(cleaned_csv, index=False)
    print(f"Wrote cleaned metadata to {cleaned_csv}")
