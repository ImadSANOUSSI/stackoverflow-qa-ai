# stackoverflow-qa-ai



## Overview
This project builds a semantic search over StackOverflow data using SentenceTransformers and a pure NumPy memmap vector store (no native compilers or extra system installs required). It serves a professional Streamlit app that answers your question with the best-matching answer from the trained index.

## Structure
- `data/` — input CSVs (Questions.csv, Answers.csv) and the trained NumPy memmap embeddings/metadata.
- `train.py` — streaming training pipeline, builds normalized embeddings into a NumPy memmap from `data/` and (optionally) deletes raw CSVs afterward.
- `app_streamlit.py` — Streamlit application that loads the index and returns the best answer with supporting context.
- `.streamlit/config.toml` — Streamlit theme (dark, professional).
- `requirements.txt` — Python dependencies.

Note: The legacy FastAPI app exists in `app/`, but after index-based workflow and data deletion, you should use the Streamlit app.

## Setup
1) Create and activate a virtual environment (recommended).
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Train the index
`train.py` streams Answers to pick the best answer per question (by Score), then streams Questions to build documents "Question + Best Answer" and encodes them with `all-MiniLM-L6-v2`. Embeddings are L2-normalized and written into a NumPy memmap.

Artifacts produced after training completes:
- `data/embeddings.dat` — NumPy memmap of shape `[N, D]` with normalized vectors
- `data/embeddings_docs.pkl` — compact metadata: question + best answer per row
- `data/vector_meta.json` — dimension/count/normalization info
- `data/sentence_transformer/` — local copy of the embedding model (saved at the end of part 4)

You have two ways to train:

### Option A) Single pass (one long run)
```bash
python train.py --batch_size 512
```
- Add `--delete_data` to remove `Questions.csv` and `Answers.csv` after a successful run.

### Option B) Split across 4 days (CPU-friendly and resumable)
We provide four scripts that split Questions into contiguous 25% chunks and write into the same `data/embeddings.dat` file. Each script shows detailed progress bars for skipping and embedding.

Run these in order (e.g., one per day):
```bash
# Day 1
python train1.py --batch_size 256

# Day 2
python train2.py --batch_size 256

# Day 3
python train3.py --batch_size 256

# Day 4 (also finalizes metadata and saves the model directory)
python train4.py --batch_size 256
```
Notes for 4-part training:
- Do not delete `data/Questions.csv` or `data/Answers.csv` until after `train4.py` finishes.
- Each part writes its slice into `data/embeddings.dat` and saves per-part metadata: `data/embeddings_docs_part{N}.pkl`.
- `train4.py` merges per-part metadata into `data/embeddings_docs.pkl`, writes `data/vector_meta.json`, and saves the embedding model to `data/sentence_transformer/`.
- You can interrupt and re-run a part; it will overwrite its own slice safely.

Avoid TensorFlow/Keras import errors (CPU-only):
If your environment tries to import TensorFlow/Keras through Transformers, set these once in your shell before running training:
```powershell
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
```
The provided training scripts already set these variables programmatically, but setting them in the shell can help keep the console clean.

Hardware/time: This streams large CSVs (multiple GB). On CPU, expect hours per part. Reduce memory pressure by using a smaller `--batch_size` (e.g., 64–256).

## Assets (excluded from Git)
Large artifacts are intentionally excluded via `.gitignore` to keep the repository small and fast to clone. You need these files locally to run the app:
- `data/embeddings.dat`
- `data/embeddings_docs.pkl`
- `data/vector_meta.json`
- Optional offline model dir: `data/sentence_transformer/`

### Get the assets
Use the downloader script to fetch artifacts from your storage (Google Drive, S3, Hugging Face, etc.). Set URLs in a `.env` file at the project root:

```env
# Required
EMBEDDINGS_DAT_URL=https://.../embeddings.dat
EMBEDDINGS_DOCS_PKL_URL=https://.../embeddings_docs.pkl
VECTOR_META_JSON_URL=https://.../vector_meta.json

# Optional (zip containing a top-level directory named sentence_transformer/)
MODEL_ZIP_URL=https://.../model.zip
```

Then run:

```bash
python scripts/download_assets.py
```

The script downloads into `data/` and prints a summary. It exits non‑zero if any core file is missing.

## Run the Streamlit app
```bash
streamlit run app_streamlit.py
```
Open the provided local URL in the browser. Enter a question; the app retrieves top matches and shows the best answer and context.

## Notes
- The vector store files must exist (`data/embeddings.dat`, `data/embeddings_docs.pkl`, `data/vector_meta.json`) before starting the app.
- If you deleted the raw CSVs, retraining requires restoring them.
- To ensure offline behavior and consistent embeddings, the app can be updated to load the local model path `data/sentence_transformer/` (currently it loads by model name and will use the local cache if present).
