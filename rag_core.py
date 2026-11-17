import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from step1_extract import load_pdf
from step2_chunk import chunk_pages

load_dotenv()

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3.6-flash"]

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),
)

# The heavy embedding library is loaded only when first needed,
# so the app page opens instantly instead of staying blank.
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_index(files):
    """files = list of (filename, file_object). Returns (chunks, embeddings)."""
    chunks = []
    for name, f in files:
        pages = load_pdf(f)
        for c in chunk_pages(pages, skip_pages=0):
            c["source"] = name
            chunks.append(c)
    if not chunks:
        raise ValueError("No readable text found. The PDF may be scanned (images only).")
    embeddings = get_embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True
    )
    return chunks, embeddings


def retrieve(question, chunks, embeddings, k=6):
    q_vec = get_embedder().encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_vec
    top_ids = np.argsort(scores)[::-1][:k]
    return [{"source": chunks[i]["source"], "page": chunks[i]["page"],
             "score": float(scores[i]), "text": chunks[i]["text"]} for i in top_ids]


def call_gemini(prompt, retries=2):
    last_error = None
    for model in MODELS:
        for _ in range(retries):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except Exception as e:
                last_error = e
                time.sleep(2)
    raise last_error


def answer(question, chunks, embeddings, k=6):
    results = retrieve(question, chunks, embeddings, k)
    context = "\n\n".join(
        f"[Source: {r['source']}, page {r['page']}]\n{r['text']}" for r in results
    )
    prompt = f"""You are a helpful assistant that answers questions about the user's uploaded documents.
Answer ONLY using the context below.
If the question asks for a summary or the main topic, summarize what the context covers.
If the answer is not in the context, say: "I could not find this in the document."
Mention the file name and page number(s) you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return call_gemini(prompt), results