import os
import sys
import json
import urllib.request
from urllib.error import URLError, HTTPError
from contextlib import closing
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import zipfile

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "sentence_transformer"

# Files we can fetch
ARTIFACTS = {
    "embeddings.dat": "EMBEDDINGS_DAT_URL",
    "embeddings_docs.pkl": "EMBEDDINGS_DOCS_PKL_URL",
    "vector_meta.json": "VECTOR_META_JSON_URL",
    # Optional: zipped model directory for offline use
    # Provide a zip containing the folder 'sentence_transformer/' at its root
    "model.zip": "MODEL_ZIP_URL",
}


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(urllib.request.urlopen(url)) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunk = 1024 * 256
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    if total:
                        pbar.update(len(buf))
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} while downloading {url}") from e
    except URLError as e:
        raise RuntimeError(f"URL error while downloading {url}: {e.reason}") from e


def _extract_model_zip(zip_path: Path, target_dir: Path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    # If the zip contained a top-level folder, try to normalize to sentence_transformer/
    # Otherwise, assume it already extracted to DATA_DIR/sentence_transformer
    if not target_dir.exists():
        # Try to find a single directory that looks like the model dir
        candidates = [p for p in DATA_DIR.iterdir() if p.is_dir() and "sentence" in p.name.lower()]
        if candidates:
            candidates[0].rename(target_dir)


def main():
    load_dotenv(BASE_DIR / ".env")  # optional

    # Read URLs from env
    urls = {name: os.getenv(env_key, "").strip() for name, env_key in ARTIFACTS.items()}

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download core artifacts
    tasks = [
        ("embeddings.dat", DATA_DIR / "embeddings.dat"),
        ("embeddings_docs.pkl", DATA_DIR / "embeddings_docs.pkl"),
        ("vector_meta.json", DATA_DIR / "vector_meta.json"),
    ]

    for fname, dest in tasks:
        url = urls[fname]
        if not url:
            print(f"[skip] {fname}: no URL set (env {ARTIFACTS[fname]})")
            continue
        if dest.exists():
            print(f"[skip] {fname}: already exists at {dest}")
            continue
        print(f"[download] {fname} from {url}")
        _download(url, dest)

    # Optional model zip
    model_zip_url = urls["model.zip"]
    if model_zip_url:
        zip_dest = DATA_DIR / "model.zip"
        if not MODEL_DIR.exists():
            print(f"[download] model.zip from {model_zip_url}")
            _download(model_zip_url, zip_dest)
            print("[extract] model.zip ->", DATA_DIR)
            _extract_model_zip(zip_dest, MODEL_DIR)
            try:
                zip_dest.unlink()
            except Exception:
                pass
        else:
            print("[skip] model directory already present at", MODEL_DIR)

    # Summary
    present = {
        "embeddings.dat": (DATA_DIR / "embeddings.dat").exists(),
        "embeddings_docs.pkl": (DATA_DIR / "embeddings_docs.pkl").exists(),
        "vector_meta.json": (DATA_DIR / "vector_meta.json").exists(),
        "sentence_transformer/": MODEL_DIR.exists(),
    }
    print("\nSummary:")
    print(json.dumps(present, indent=2))

    # Exit non-zero if the 3 core artifacts are missing
    missing_core = [k for k in ["embeddings.dat", "embeddings_docs.pkl", "vector_meta.json"] if not present[k]]
    if missing_core:
        print("\nWarning: missing core artifacts:", ", ".join(missing_core))
        print("Set their URLs in .env or environment variables and rerun.")
        sys.exit(1)


if __name__ == "__main__":
    main()
