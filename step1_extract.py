import os
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "sample.pdf")

OCR_WORKERS = 4   # pages read at the same time


def load_pdf(path, ocr_fallback=True):
    reader = PdfReader(path)
    total = len(reader.pages)
    texts = {}
    scanned = {}   # page number -> single-page PDF bytes

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 30 and ocr_fallback:
            from ocr import page_to_bytes
            scanned[i] = page_to_bytes(page)
        texts[i] = text

    if scanned:
        from ocr import ocr_bytes
        print(f"{len(scanned)} of {total} pages need OCR ({OCR_WORKERS} at a time)...", flush=True)
        done = 0

        def work(item):
            num, data = item
            return num, ocr_bytes(data)

        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
            for num, text in pool.map(work, scanned.items()):
                texts[num] = text
                done += 1
                print(f"  OCR done: {done}/{len(scanned)}", flush=True)

    return [{"page": i, "text": texts[i]} for i in range(1, total + 1)]


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    print(f"Total pages: {len(pages)}")
    print("--- Page 1 preview ---")
    print(pages[0]["text"][:500])