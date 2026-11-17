from step1_extract import load_pdf, PDF_PATH

def chunk_pages(pages, chunk_size=800, overlap=150, skip_pages=3):
    chunks = []
    for page in pages:
        if page["page"] <= skip_pages:
            continue
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]
        current, length = [], 0
        for line in lines:
            if length + len(line) > chunk_size and current:
                chunks.append({"page": page["page"], "text": "\n".join(current)})
                # keep the last few lines as overlap
                tail, tail_len = [], 0
                for l in reversed(current):
                    if tail_len + len(l) > overlap:
                        break
                    tail.insert(0, l)
                    tail_len += len(l)
                current, length = tail, tail_len
            current.append(line)
            length += len(line)
        if current:
            chunks.append({"page": page["page"], "text": "\n".join(current)})
    return chunks

if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)
    print(f"Total chunks: {len(chunks)}")
    print("\n--- Chunk 1 ---")
    print(f"Page: {chunks[0]['page']}")
    print(chunks[0]["text"])