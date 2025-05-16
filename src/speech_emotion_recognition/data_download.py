import zipfile
import urllib.request
from pathlib import Path
import sys

def download_dataset(url: str, cfg: dict) -> None:
    """
    Downloads the dataset ZIP from `url` with a progress bar,
    then extracts it into cfg['paths']['raw_data_dir'].
    """
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir.parent / "dataset.zip"
    print(f"Downloading dataset from:\n  {url}\n→ {zip_path}")

    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            if downloaded > total_size:
                downloaded = total_size
            bar_len = 40
            filled_len = int(bar_len * downloaded / total_size)
            bar = "#" * filled_len + "-" * (bar_len - filled_len)
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r  [{bar}] {percent:6.2f}%")
        else:
            sys.stdout.write(f"\r  Downloaded {downloaded} bytes")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook)
        sys.stdout.write("\r" + " " * 60 + "\r")
        print("Download complete.")
    except Exception as e:
        print(f"\nError downloading: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting archive to {raw_dir} …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        zip_path.unlink()

    print("Dataset downloaded and extracted.")
