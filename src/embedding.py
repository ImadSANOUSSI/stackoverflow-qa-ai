# src/embedding.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_PATH = "data/source.csv"
VECTOR_STORE_PATH = "data/faiss_index"
TEXT_COLUMN = "text"  # Change if your column name is different

def embed_documents():
    df = pd.read_csv(DATA_PATH)
    docs = df[TEXT_COLUMN].dropna().tolist()

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(docs, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save index + docs
    faiss.write_index(index, f"{VECTOR_STORE_PATH}.index")
    with open(f"{VECTOR_STORE_PATH}_docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Indexed {len(docs)} documents.")

if __name__ == "__main__":
    embed_documents()
