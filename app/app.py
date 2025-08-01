import os
import pandas as pd
import numpy as np
import faiss
import subprocess
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
# No OpenAI key needed

# === FastAPI setup ===
app = FastAPI()
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# === Path setup ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
index = faiss.read_index(os.path.join(APP_DIR, "index.index"))

# === Load datasets ===
questions_df = pd.read_csv(os.path.join(DATA_DIR, "Questions.csv"), encoding='latin1')

answers_df = pd.read_csv(os.path.join(DATA_DIR, "Answers.csv"), encoding='latin1')
questions_df['Id'] = questions_df['Id'].astype(int)
answers_df['ParentId'] = answers_df['ParentId'].astype(int)

# === Load embedding model ===
model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())  # Should print 384

# === Function: Generate response using Ollama CLI llama3.2 ===
def generate_with_llama(context: str, question: str) -> str:
    prompt = f"""
Tu es un assistant intelligent. Voici une question posée par un utilisateur, ainsi que des extraits pertinents tirés de StackOverflow :

Context:
{context}

Question: {question}

Réponds de manière claire, concise et utile en français.
"""
    try:
        process = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        if process.returncode != 0:
            return f"Erreur Ollama: {process.stderr.decode('utf-8')}"

        answer = process.stdout.decode("utf-8").strip()
        return answer

    except Exception as e:
        return f"Erreur Ollama exception: {str(e)}"

# === Function: Retrieve relevant Q&A and generate answer ===
def get_rag_answer(user_question: str):
    query_vec = model.encode([user_question])
    query_vec = np.array(query_vec).astype("float32")

    distances, indices = index.search(query_vec, k=3)
    matched_questions = []
    matched_answers = []

    for idx in indices[0]:
        q_row = questions_df.iloc[idx]
        q_id = q_row["Id"]
        matched_questions.append(q_row["Body"])

        related_answers = answers_df[answers_df['ParentId'] == q_id]
        if not related_answers.empty:
            best_ans = related_answers.loc[related_answers['Score'].idxmax()]
            matched_answers.append(best_ans["Body"])
        else:
            matched_answers.append("Pas de réponse disponible.")

    context = ""
    for q, a in zip(matched_questions, matched_answers):
        context += f"Question: {q}\nRéponse: {a}\n---\n"

    generated_answer = generate_with_llama(context, user_question)

    return matched_questions, matched_answers, generated_answer

# === FastAPI Routes ===
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def answer_question(request: Request, question: str = Form(...)):
    matched_questions, matched_answers, generated_answer = get_rag_answer(question)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_question": question,
            "matched_question": matched_questions[0] if matched_questions else "Aucune question similaire trouvée.",
            "answer": generated_answer,
            "retrieved_context": zip(matched_questions, matched_answers),
        },
    )
