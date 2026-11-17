import io
import streamlit as st
from rag_core import build_index, answer

st.set_page_config(page_title="Rayquora AI", page_icon="⚡", layout="centered")


@st.cache_data(show_spinner=False)
def cached_build(files_data):
    """Index the PDFs once. Re-uploading the same files loads instantly."""
    files = [(name, io.BytesIO(data)) for name, data in files_data]
    return build_index(files)


# ---------- FUTURISTIC STYLE ----------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(0,245,212,0.12), transparent 40%),
                radial-gradient(circle at 85% 20%, rgba(155,93,229,0.15), transparent 45%),
                radial-gradient(circle at 50% 100%, rgba(241,91,181,0.10), transparent 50%),
                #0B0F1A;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.neon-title {
    font-family: 'Orbitron', 'Segoe UI', sans-serif; font-weight: 800; font-size: 2.6rem;
    text-align: center;
    background: linear-gradient(90deg, #00F5D4, #9B5DE5, #F15BB5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 14px rgba(0,245,212,0.45)); margin-bottom: 0;
}
.neon-sub {
    text-align: center; color: #8FA3BF; letter-spacing: 3px; font-size: 0.8rem;
    text-transform: uppercase; margin-bottom: 1.5rem;
}
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(0,245,212,0.18);
    border-radius: 16px; padding: 14px 18px; margin-bottom: 12px;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(155,93,229,0.4); border-radius: 12px;
    background: rgba(155,93,229,0.06);
}
section[data-testid="stSidebar"] {
    background: rgba(18,26,43,0.85); border-right: 1px solid rgba(0,245,212,0.2);
}
.stButton > button {
    background: linear-gradient(90deg, #00F5D4, #9B5DE5); color: #0B0F1A;
    font-weight: 600; border: none; border-radius: 10px;
}
.chip {
    display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
    border-radius: 999px; border: 1px solid #00F5D4; color: #00F5D4; font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="neon-title">⚡ RAYQUORA AI</p>', unsafe_allow_html=True)
st.markdown('<p class="neon-sub">Chat with your documents · RAG Powered</p>', unsafe_allow_html=True)

# ---------- STATE ----------
for key, default in [("messages", []), ("chunks", None),
                     ("embeddings", None), ("doc_key", None),
                     ("uploader_key", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- SIDEBAR: UPLOAD ----------
with st.sidebar:
    st.markdown("### 📂 Your documents")
    uploaded = st.file_uploader("Upload one or more PDFs", type="pdf",
                                accept_multiple_files=True,
                                key=f"uploader_{st.session_state.uploader_key}")

    if uploaded:
        doc_key = tuple((f.name, f.size) for f in uploaded)
        if doc_key != st.session_state.doc_key:
            with st.spinner("Reading and indexing your PDFs (scanned pages take a few minutes)..."):
                try:
                    chunks, embeddings = cached_build(
                        tuple((f.name, f.getvalue()) for f in uploaded)
                    )
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.doc_key = doc_key
                    st.session_state.messages = []
                except Exception as e:
                    st.error(str(e))
        if st.session_state.chunks:
            st.success(f"Ready: {len(uploaded)} file(s), {len(st.session_state.chunks)} chunks")
    else:
        st.session_state.chunks = None
        st.session_state.doc_key = None

    st.markdown("---")
    st.markdown("**Embeddings:** MiniLM-L6-v2")
    st.markdown("**Retrieval:** Top-6 cosine similarity")
    st.markdown("**LLM:** Gemini")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Remove files & reset"):
        st.session_state.uploader_key += 1
        st.session_state.messages = []
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.doc_key = None
        st.rerun()

# ---------- CHAT ----------
def show_sources(results):
    labels = sorted({(r["source"], r["page"]) for r in results})
    chips = "".join(f'<span class="chip">{s} · p.{p}</span>' for s, p in labels)
    st.markdown(chips, unsafe_allow_html=True)
    with st.expander("View source text"):
        for r in results:
            st.markdown(f"**{r['source']} · page {r['page']}** · match {r['score']:.2f}")
            st.text(r["text"][:300])

if st.session_state.chunks is None:
    st.info("👈 Upload a PDF in the sidebar to start chatting with it.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask Rayquora about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Rayquora is thinking..."):
                try:
                    text, results = answer(question, st.session_state.chunks,
                                           st.session_state.embeddings)
                except Exception as e:
                    text, results = f"The AI service is busy right now. Please try again. ({str(e)[:80]})", []
            st.markdown(text)
            if results:
                show_sources(results)

        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": results}
        )