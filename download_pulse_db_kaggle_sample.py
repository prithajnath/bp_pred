from tempfile import NamedTemporaryFile
import urllib.request
import zipfile
import os

url = "https://www.kaggle.com/api/v1/datasets/download/weinanwangrutgers/pulsedb-balanced-training-and-testing"
extract_dir = os.path.join("data", "pulsedb-balanced-training-and-testing")


def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(
            f"\r{pct:.1f}% ({downloaded // 1024**2} MB / {total_size // 1024**2} MB)",
            end="",
        )


with NamedTemporaryFile(mode="wb", delete=True) as f:
    print("Downloading...")
    urllib.request.urlretrieve(url, f.name, reporthook=progress)
    print(f"\nSaved to {f.name}")

    print("Extracting...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(f.name, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")
