import os
import pickle
import time
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import json
import re
from html import unescape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMBED_PATH = os.path.join(DATA_DIR, 'embeddings.dat')
DOCS_META_PATH = os.path.join(DATA_DIR, 'embeddings_docs.pkl')
VECTOR_META_PATH = os.path.join(DATA_DIR, 'vector_meta.json')
MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource(show_spinner=False)
def load_resources():
    # Prefer offline/local model
    local_model_dir = os.path.join(DATA_DIR, 'sentence_transformer')
    model = None
    try:
        if os.path.isdir(local_model_dir):
            model = SentenceTransformer(local_model_dir)
        else:
            # Try using HF cache only (no network)
            try:
                model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            except TypeError:
                # Older versions may not support local_files_only directly; rely on env var
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            "Cannot load embedding model offline. Copy a downloaded SentenceTransformer to 'data/sentence_transformer/' or ensure the model is present in the local HuggingFace cache and retry."
        ) from e
    # Reconstruct vector_meta.json if missing
    if not os.path.exists(VECTOR_META_PATH):
        # Load docs metadata to get count
        with open(DOCS_META_PATH, 'rb') as f:
            docs_meta = pickle.load(f)
        count = int(len(docs_meta))
        # Infer dim from embeddings.dat size (float32 = 4 bytes)
        file_bytes = os.path.getsize(EMBED_PATH)
        total_floats = file_bytes // 4
        if count == 0 or total_floats % count != 0:
            raise RuntimeError("Unable to infer embedding dimension from embeddings.dat and docs metadata.")
        dim = int(total_floats // count)
        vec_meta = {'dim': dim, 'count': count, 'normalized': True}
        # Persist for future runs
        with open(VECTOR_META_PATH, 'w', encoding='utf-8') as f:
            json.dump(vec_meta, f)
    else:
        with open(VECTOR_META_PATH, 'r', encoding='utf-8') as f:
            vec_meta = json.load(f)
        dim = int(vec_meta['dim'])
        count = int(vec_meta['count'])
        # Load docs metadata
        with open(DOCS_META_PATH, 'rb') as f:
            docs_meta = pickle.load(f)

    emb_mm = np.memmap(EMBED_PATH, dtype='float32', mode='r', shape=(count, dim))
    return model, emb_mm, docs_meta, vec_meta


def search(query: str, top_k: int = 5, chunk_size: int = 200_000):
    model, emb_mm, docs_meta, vec_meta = load_resources()
    q = model.encode([query], convert_to_numpy=True).astype('float32')
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    qv = q[0]

    n = emb_mm.shape[0]
    best_scores = None  # cosine similarity
    best_indices = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(emb_mm[start:end])  # shape (m, d), already normalized
        sims = chunk @ qv  # cosine similarity
        if best_scores is None:
            # initialize
            if end - start >= top_k:
                idx = np.argpartition(sims, -top_k)[-top_k:]
            else:
                idx = np.arange(end - start)
            best_scores = sims[idx]
            best_indices = (idx + start)
        else:
            # merge with existing top_k
            if end - start >= top_k:
                idx = np.argpartition(sims, -top_k)[-top_k:]
            else:
                idx = np.arange(end - start)
            cand_scores = np.concatenate([best_scores, sims[idx]])
            cand_indices = np.concatenate([best_indices, idx + start])
            if cand_scores.size > top_k:
                keep = np.argpartition(cand_scores, -top_k)[-top_k:]
                best_scores = cand_scores[keep]
                best_indices = cand_indices[keep]
            else:
                best_scores = cand_scores
                best_indices = cand_indices

    # sort descending by similarity
    order = np.argsort(-best_scores)
    best_indices = best_indices[order]
    best_scores = best_scores[order]

    results = []
    for i, sim in zip(best_indices, best_scores):
        meta = docs_meta[int(i)]
        results.append({
            'score': float(1.0 - sim),  # convert to cosine distance for display
            'question_id': meta.get('question_id'),
            'question_body': meta.get('question_body', ''),
            'best_answer_body': meta.get('best_answer_body', ''),
        })
    return results


# UI
st.set_page_config(page_title='StackOverflow QA Assistant', page_icon='💬', layout='wide')

st.title('💬 StackOverflow QA Assistant')

st.write('Ask a technical question. I will search a trained knowledge base and provide the best-matching answer.')

with st.sidebar:
    st.header('Settings')
    top_k = st.slider('Results to retrieve', 1, 10, 5)

query = st.text_input('Your question', placeholder='e.g., How to sort a dictionary by value in Python?')

col1, col2 = st.columns([1, 4])
with col1:
    submitted = st.button('Search')
with col2:
    st.caption('Powered by SentenceTransformers + NumPy (cosine search)')

# Simple HTML -> plain text cleaner (remove tags like <p>, <li>, etc.)
def clean_html(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = unescape(text)
    # Common line-break preserving replacements
    t = re.sub(r'<\s*br\s*/?>', '\n', t, flags=re.IGNORECASE)
    t = re.sub(r'</\s*p\s*>', '\n\n', t, flags=re.IGNORECASE)
    t = re.sub(r'<\s*p\s*>', '', t, flags=re.IGNORECASE)
    t = re.sub(r'<\s*li\s*>', '- ', t, flags=re.IGNORECASE)
    t = re.sub(r'</\s*li\s*>', '\n', t, flags=re.IGNORECASE)
    # Drop any remaining tags
    t = re.sub(r'<[^>]+>', '', t)
    # Normalize whitespace
    t = re.sub(r'\r\n|\r', '\n', t)
    t = re.sub(r'\n\n\n+', '\n\n', t).strip()
    return t

if submitted and query.strip():
    with st.spinner('Searching knowledge base...'):
        t0 = time.time()
        results = search(query, top_k=top_k)
        dt = time.time() - t0

    if not results:
        st.warning('No results found.')
    else:
        # Best answer
        best = results[0]
        st.subheader('✅ Best Answer')
        st.markdown(clean_html(best['best_answer_body']) or 'No answer available.')

        st.caption(f"Retrieved in {dt:.2f}s | Query: {query}")

        # Contextual results
        with st.expander('See retrieved context', expanded=False):
            for r in results:
                st.markdown('---')
                st.markdown('**Question:**')
                st.markdown(clean_html(r['question_body']) or '*No question text*')
                st.markdown('**Top Answer:**')
                st.markdown(clean_html(r['best_answer_body']) or '*No answer available*')
                st.caption(f"Distance: {r['score']:.4f} | Question ID: {r['question_id']}")
else:
    st.info('Enter a question and click Search to get started.')
