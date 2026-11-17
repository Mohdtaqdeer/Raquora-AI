import os, io, time, hashlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfWriter

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),
)

OCR_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]

PROMPT = ("Transcribe all the text on this page exactly as written, including "
          "handwriting, formulas and tables. Output only the text, with no comments. "
          "If the page has no text, output nothing.")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def page_to_bytes(page_obj):
    """Turn one pypdf page into a small one-page PDF (bytes)."""
    writer = PdfWriter()
    writer.add_page(page_obj)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def ocr_bytes(data):
    """OCR one single-page PDF. Results are cached on disk by content hash."""
    key = hashlib.sha1(data).hexdigest()
    cache_file = os.path.join(CACHE_DIR, key + ".txt")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return f.read()

    for model in OCR_MODELS:
        for _ in range(2):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=data, mime_type="application/pdf"), PROMPT],
                )
                text = (r.text or "").strip()
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
            except Exception:
                time.sleep(2)
    return ""   # failed: not cached, so it is retried next time