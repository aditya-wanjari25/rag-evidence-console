from dotenv import load_dotenv
load_dotenv()

import argparse
import os

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
            # Simple loader for v0: text files only
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({
                    "text": f.read(),
                    "source": filename
                })
    return documents


def build_pipeline(cfg: dict) -> RAGPipeline:
    # Chunker
    chunk_cfg = cfg["chunking"]
    chunker = FixedSizeChunker(
        chunk_size=int(chunk_cfg["chunk_size"]),
        overlap=int(chunk_cfg["overlap"])
    )

    # Retriever
    ret_cfg = cfg["retrieval"]
    retriever = OpenAIFaissRetriever(
        embedding_model=ret_cfg.get("embedding_model", "text-embedding-3-small")
    )

    # Generator
    llm_cfg = cfg["llm"]
    generator = OpenAIGenerator(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0))
    )

    return RAGPipeline(chunker, retriever, generator)


def main():
    parser = argparse.ArgumentParser(description="RAG Evidence & Trust Console CLI")

    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path", nargs="?", help="Path to documents folder (optional)")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--docs", help="Docs directory override (optional)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg)

    docs_dir_default = cfg["paths"].get("docs_dir", "docs")
    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--yes", action="store_true", help="Confirm reset")


    if args.command == "ingest":
        docs_dir = args.path or docs_dir_default
        docs = load_documents(docs_dir)
        pipeline.ingest(docs)
        artifacts_dir = cfg["paths"].get("artifacts_dir", "artifacts")
        pipeline.retriever.save(artifacts_dir)
        print(f"Ingested {len(docs)} doc(s). Saved index to: {artifacts_dir}")


    elif args.command == "query":
        # v0 behavior: build index fresh each run unless we add persistence later
        docs_dir = args.docs or docs_dir_default
        docs = load_documents(docs_dir)
        pipeline.ingest(docs)
        artifacts_dir = cfg["paths"].get("artifacts_dir", "artifacts")
        pipeline.retriever.load(artifacts_dir)


        k = int(cfg["retrieval"].get("top_k", 5))
        retrieved = pipeline.retriever.retrieve(args.question, k=k)
        result = pipeline.generator.generate(args.question, retrieved)

        print("\n=== Retrieved Chunks ===")
        for i, r in enumerate(retrieved, start=1):
            src = r.get("metadata", {}).get("source", "unknown")
            print(f"\n[{i}] score={r['score']:.4f} source={src} chunk_id={r['chunk_id']}")
            print(r["text"][:400], "..." if len(r["text"]) > 400 else "")

        print("\n=== Answer ===")
        print(result["answer"])

    elif args.command == "reset":
        if not args.yes:
            print("Add --yes to confirm reset.")
            return
        import shutil
        artifacts_dir = cfg["paths"].get("artifacts_dir", "artifacts")
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)
            print(f"Deleted artifacts directory: {artifacts_dir}")
        else:
            print("No artifacts directory found. Nothing to reset.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
