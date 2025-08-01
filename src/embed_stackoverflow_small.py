import pandas as pd
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import os
import ollama  
import json

# === CONFIG ===
DATA_PATH = "data/Questions.csv"
VECTOR_STORE_PATH = "data/faiss_index"
TEXT_COLUMN = "Body"
NUM_ROWS = 3000 * 60
BATCH_SIZE = 256

# === NOM DU MODELE OLLAMA ===
OLLAMA_MODEL = "nomic-embed-text"  

def embed_with_ollama(texts):
    results = []
    for text in texts:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        embedding = response["embedding"]
        results.append(embedding)
    return np.array(results).astype("float32")


def embed_documents():
    print("📥 Loading data...")

    with open(DATA_PATH, encoding='latin1') as f:
        df = pd.read_csv(f, usecols=["Id", "Body", "Score"])

    docs = df[TEXT_COLUMN].dropna().astype(str).head(NUM_ROWS).tolist()
    print(f"✅ {len(docs)} documents will be embedded with Ollama model '{OLLAMA_MODEL}'.")

    embeddings = []
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="🔄 Embedding"):
        batch = docs[i:i + BATCH_SIZE]
        emb = embed_with_ollama(batch)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    print("🧠 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    faiss.write_index(index, f"{VECTOR_STORE_PATH}.index")
    with open(f"{VECTOR_STORE_PATH}_docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Done! Saved index for {len(docs)} documents to:")
    print(f"   📄 {VECTOR_STORE_PATH}.index")
    print(f"   📝 {VECTOR_STORE_PATH}_docs.pkl")


if __name__ == "__main__":
    embed_documents()
