# WisdomLlm

A FastAPI-based Question & Answer web app using StackOverflow data, semantic search with FAISS, and LLM-powered answer generation.

## Project Structure

```
WisdomLlm/
│
├── app/
│   ├── app.py              # Main FastAPI app
│   ├── index.index         # FAISS index file
│   └── templates/
│       └── index.html      # Jinja2 template
│
├── data/
│   ├── Answers.csv
│   ├── Questions.csv
│   ├── Tags.csv
│   ├── source.csv
│   ├── faiss_index.index
│   └── faiss_index_docs.pkl
│
├── src/
│   ├── create_index.py     # Script to create FAISS index
│   ├── embedding.py
│   └── embed_stackoverflow_small.py
│
└── .venv/                  # Python virtual environment (optional)
```

## Setup

1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place your StackOverflow CSV files in the `data/` directory.

3. **Create the FAISS index**
   ```sh
   python src/create_index.py
   ```
   This will generate `app/index.index` using random data (replace with real embeddings for production).

4. **Run the FastAPI app**
   ```sh
   uvicorn app.app:app --reload
   ```
   Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Usage

- Enter a question in the web form.
- The app retrieves the most relevant StackOverflow Q&A using semantic search (FAISS + SentenceTransformer).
- The context is sent to an LLM (via Ollama) to generate a concise answer in French.

## Notes

- The default FAISS index is built with random vectors for testing. For real use, generate embeddings from your actual questions.
- The app expects `Questions.csv` and `Answers.csv` to have columns: `Id`, `Body`, `ParentId`, `Score`, etc.
- Ollama must be installed and accessible from the command line for LLM answer generation.

## License

MIT License