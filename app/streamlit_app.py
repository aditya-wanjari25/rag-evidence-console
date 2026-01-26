from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from config.load import load_config
from core.chunking.fixed import FixedSizeChunker
from core.retrieval.vector import OpenAIFaissRetriever
from core.generation.llm import OpenAIGenerator
from core.pipeline import RAGPipeline


def load_documents(path: str):
    documents = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            # v1: simple text loader only (we’ll add PDF later)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({"text": f.read(), "source": filename})
    return documents


def build_pipeline(cfg: dict) -> RAGPipeline:
    chunk_cfg = cfg["chunking"]
    chunker = FixedSizeChunker(
        chunk_size=int(chunk_cfg["chunk_size"]),
        overlap=int(chunk_cfg["overlap"]),
    )

    ret_cfg = cfg["retrieval"]
    retriever = OpenAIFaissRetriever(
        embedding_model=ret_cfg.get("embedding_model", "text-embedding-3-small")
    )

    llm_cfg = cfg["llm"]
    generator = OpenAIGenerator(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0)),
    )

    return RAGPipeline(chunker, retriever, generator)


def ensure_index_loaded(pipeline: RAGPipeline, artifacts_dir: str) -> bool:
    try:
        pipeline.retriever.load(artifacts_dir)
        return True
    except Exception as e:
        st.warning(f"Index not loaded yet. Run ingestion. Details: {e}")
        return False


def main():
    st.set_page_config(page_title="RAG Evidence & Trust Console", layout="wide")
    st.title("RAG Evidence & Trust Console — Version 1 (Transparent RAG)")

    cfg = load_config("config/default.yaml")
    pipeline = build_pipeline(cfg)

    docs_dir = cfg["paths"].get("docs_dir", "docs")
    artifacts_dir = cfg["paths"].get("artifacts_dir", "artifacts")
    top_k_default = int(cfg["retrieval"].get("top_k", 5))

    with st.sidebar:
        st.header("Controls")
        st.caption("Version 1: persistent index + evidence view")

        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=20, value=top_k_default)

        st.divider()
        st.subheader("Index lifecycle")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("(Re)Ingest docs", use_container_width=True):
                docs = load_documents(docs_dir)
                if len(docs) == 0:
                    st.error(f"No documents found in {docs_dir}")
                else:
                    pipeline.ingest(docs)
                    os.makedirs(artifacts_dir, exist_ok=True)
                    pipeline.retriever.save(artifacts_dir)
                    st.success(f"Ingested {len(docs)} docs and saved index to {artifacts_dir}")

        with col_b:
            if st.button("Load index", use_container_width=True):
                ok = ensure_index_loaded(pipeline, artifacts_dir)
                if ok:
                    st.success(f"Loaded index from {artifacts_dir}")

        st.divider()
        st.subheader("Paths")
        st.text(f"docs_dir: {docs_dir}")
        st.text(f"artifacts_dir: {artifacts_dir}")

    # main layout
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Ask a question")
        question = st.text_area("Question", placeholder="e.g., What is RAG?", height=110)
        ask = st.button("Run query", type="primary", use_container_width=True)

    with right:
        st.subheader("Answer")
        answer_box = st.empty()

    if ask:
        if not question.strip():
            st.error("Please enter a question.")
            return

        # Load index (required for querying)
        ok = ensure_index_loaded(pipeline, artifacts_dir)
        if not ok:
            return

        # Retrieve
        retrieved = pipeline.retriever.retrieve(question, k=top_k)

        # Generate
        result = pipeline.generator.generate(question, retrieved)

        answer_box.markdown(result["answer"])

        st.divider()
        st.subheader("Evidence (Retrieved Chunks)")

        # Render chunks as expandable cards
        for i, r in enumerate(retrieved, start=1):
            src = r.get("metadata", {}).get("source", "unknown")
            title = f"[{i}] score={r['score']:.4f} | source={src} | chunk_id={r['chunk_id']}"
            with st.expander(title, expanded=(i == 1)):
                st.write(r["text"])

    st.caption("Tip: If you changed docs, click “(Re)Ingest docs” once, then query multiple times.")


if __name__ == "__main__":
    main()
