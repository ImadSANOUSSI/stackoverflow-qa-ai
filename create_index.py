# create_index.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import os

def create_faiss_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = model.get_sentence_embedding_dimension()

    # Load questions
    questions_df = pd.read_csv(os.path.join("data", "Questions.csv"), encoding='latin1')
    questions = questions_df["Body"].astype(str).tolist()

    # Compute embeddings
    embeddings = model.encode(questions, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_path = os.path.join("app", "index.index")
    faiss.write_index(index, index_path)
    print(f"Index created and saved at {index_path}")

if __name__ == "__main__":
    create_faiss_index()
