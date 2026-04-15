#!/usr/bin/env python3
"""
Build a local FAISS vector database from one or more PDF files.

Improvements in this version
- Adds structured logging
- Cleans noisy PDF text such as (cid:xx), broken whitespace, and repeated references
- Uses page-level cleanup before chunking
- Uses smaller default chunks for better retrieval quality
- Optionally skips very noisy chunks
- Persists and reuses a FAISS index when the input file set is unchanged
- Supports a smoke-test query after build

Usage
  python scripts/prebuilt_db.py ./data/papers
  python scripts/prebuilt_db.py "./data/papers/*.pdf"
  python scripts/prebuilt_db.py ./data/papers --query "hybrid bonding yield"
  python scripts/prebuilt_db.py ./data/papers --force-rebuild
  python scripts/prebuilt_db.py ./data/papers --verbose

"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger("prebuilt_db")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector DB from PDF files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for text splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for text splitting.",
    )
    parser.add_argument(
        "--cache-root",
        default=".cache/embeddings",
        help="Directory for embedding cache.",
    )
    parser.add_argument(
        "--index-root",
        default=".cache/faiss_index",
        help="Directory for FAISS index output.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional test query to run after build.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k results for the optional test query.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cached metadata and rebuild the index.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum number of characters required to keep a cleaned page/chunk.",
    )
    return parser.parse_args()


def expand_pdf_paths(items: Iterable[str]) -> list[Path]:
    logger.info("Scanning input paths...")
    pdfs: list[Path] = []

    for item in items:
        path = Path(item)
        logger.debug("Checking input: %s", item)

        if path.is_dir():
            found = sorted(p for p in path.rglob("*.pdf") if p.is_file())
            logger.info("Directory found: %s (%d pdfs)", path, len(found))
            pdfs.extend(found)
            continue

        if path.is_file():
            if path.suffix.lower() != ".pdf":
                logger.warning("Skipping non-PDF file: %s", path)
                continue
            logger.info("PDF file found: %s", path)
            pdfs.append(path)
            continue

        matches = [Path(m) for m in glob.glob(item, recursive=True)]
        if matches:
            logger.info("Glob matched %d path(s): %s", len(matches), item)
            for m in matches:
                if m.is_file() and m.suffix.lower() == ".pdf":
                    pdfs.append(m)
                elif m.is_dir():
                    found = sorted(p for p in m.rglob("*.pdf") if p.is_file())
                    pdfs.extend(found)
        else:
            logger.warning("No matches found for: %s", item)

    seen = set()
    unique: list[Path] = []
    for p in sorted(pdfs):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    logger.info("Total unique PDFs found: %d", len(unique))
    return unique


def fingerprint_files(pdf_paths: list[Path]) -> str:
    logger.info("Generating fingerprint for %d PDF file(s)...", len(pdf_paths))
    hasher = hashlib.sha256()

    for path in sorted(pdf_paths):
        stat = path.stat()
        hasher.update(str(path.resolve()).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))

    fingerprint = hasher.hexdigest()
    logger.debug("Fingerprint: %s", fingerprint)
    return fingerprint


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove common broken PDF extraction artifacts
    text = re.sub(r"\(cid:\d+\)", " ", text)

    # Normalize Unicode dashes and quotes roughly
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\xa0", " ")

    # Remove page headers/footers-like isolated page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)

    # Fix line breaks inside words: "Wafer-to-\nWafer" -> "Wafer-to-Wafer"
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)

    # Join lines that were broken mid-sentence
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove some obviously noisy reference-only tails
    text = re.sub(r"\bpp\.\s*\d+\s*-\s*\d+\s*\(\d{4}\)", " ", text)
    text = re.sub(r"\(\d{4}\)\s*$", " ", text)

    return text.strip()


def noise_score(text: str) -> float:
    if not text:
        return 1.0

    weird = len(re.findall(r"[^\w\s.,;:!?()\[\]\-/%+\"']", text))
    total = max(len(text), 1)
    return weird / total


def looks_too_noisy(text: str, min_chars: int) -> bool:
    if len(text.strip()) < min_chars:
        return True

    # Excessive formula/symbol noise
    if noise_score(text) > 0.18:
        return True

    # Too few alphabetic characters relative to total length
    alpha = len(re.findall(r"[A-Za-z가-힣]", text))
    if alpha / max(len(text), 1) < 0.25:
        return True

    return False


def load_documents(pdf_paths: list[Path], min_chars: int):
    logger.info("Loading PDF documents...")
    docs: list[Document] = []
    skipped_pages = 0

    for i, pdf_path in enumerate(pdf_paths, start=1):
        logger.info("[%d/%d] Loading: %s", i, len(pdf_paths), pdf_path)
        loader = PDFPlumberLoader(str(pdf_path))
        loaded = loader.load()
        logger.info("Loaded %d page document(s) from %s", len(loaded), pdf_path.name)

        kept_for_this_pdf = 0

        for page_doc in loaded:
            page_doc.metadata = dict(page_doc.metadata or {})
            page_doc.metadata["source"] = str(pdf_path)
            page_doc.metadata["filename"] = pdf_path.name

            raw_text = page_doc.page_content or ""
            cleaned = clean_text(raw_text)

            if looks_too_noisy(cleaned, min_chars=min_chars):
                skipped_pages += 1
                logger.debug(
                    "Skipping noisy/short page | file=%s | page=%s",
                    pdf_path.name,
                    page_doc.metadata.get("page", "?"),
                )
                continue

            page_doc.page_content = cleaned
            docs.append(page_doc)
            kept_for_this_pdf += 1

        logger.info(
            "Kept %d cleaned page(s) from %s",
            kept_for_this_pdf,
            pdf_path.name,
        )

    logger.info("Total cleaned document pages kept: %d", len(docs))
    logger.info("Total skipped noisy/short pages: %d", skipped_pages)
    return docs


def split_documents(documents, chunk_size: int, chunk_overlap: int, min_chars: int):
    logger.info(
        "Splitting documents into chunks (chunk_size=%d, overlap=%d)...",
        chunk_size,
        chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
    raw_chunks = splitter.split_documents(documents)
    logger.info("Generated %d raw chunk(s).", len(raw_chunks))

    cleaned_chunks: list[Document] = []
    skipped_chunks = 0

    for i, chunk in enumerate(raw_chunks):
        text = clean_text(chunk.page_content)

        if looks_too_noisy(text, min_chars=min_chars):
            skipped_chunks += 1
            continue

        chunk.page_content = text
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_id"] = len(cleaned_chunks)
        chunk.metadata["char_len"] = len(text)
        cleaned_chunks.append(chunk)

    logger.info("Kept %d cleaned chunk(s).", len(cleaned_chunks))
    logger.info("Skipped %d noisy/short chunk(s).", skipped_chunks)
    return cleaned_chunks


def load_or_build_index(
    chunks,
    embeddings,
    index_root: Path,
    metadata_path: Path,
    fingerprint: str,
    input_files: list[Path],
    force_rebuild: bool,
):
    logger.info("Preparing vector index...")
    logger.info("Index path: %s", index_root)

    if not force_rebuild and index_root.exists() and metadata_path.exists():
        try:
            logger.info("Existing metadata found. Checking fingerprint...")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                logger.info("Fingerprint matched. Reusing existing FAISS index.")
                return FAISS.load_local(
                    str(index_root),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            logger.info("Fingerprint changed. Rebuilding FAISS index.")
        except Exception as exc:
            logger.warning("Could not reuse existing index: %s", exc)

    logger.info("Building new FAISS index from %d chunk(s)...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_root.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_root))

    build_metadata = {
        "fingerprint": fingerprint,
        "files": [str(p.resolve()) for p in input_files],
        "num_chunks": len(chunks),
    }
    metadata_path.write_text(
        json.dumps(build_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("FAISS index saved.")
    logger.info("Metadata saved to: %s", metadata_path)
    return vectorstore


def run_query(vectorstore: FAISS, query: str, top_k: int) -> None:
    logger.info("Running smoke test query: %s", query)
    results = vectorstore.similarity_search(query, k=top_k)
    logger.info("Retrieved %d result(s).", len(results))

    if not results:
        logger.warning("No results found for query.")
        return

    for idx, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        filename = doc.metadata.get("filename", Path(source).name)
        snippet = " ".join(doc.page_content.split())[:350]

        logger.info("Result %d | %s | page=%s", idx, filename, page)
        print(f"\n[{idx}] {filename} | page={page}")
        print(f"source: {source}")
        print(snippet)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("Starting PDF -> Vector DB build process...")
    load_dotenv()
    logger.info(".env loaded.")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set.")
        logger.error("Add it to your environment or .env file.")
        return 1

    logger.info("OPENAI_API_KEY detected.")

    pdf_paths = expand_pdf_paths(args.inputs)
    if not pdf_paths:
        logger.error("No PDF files found.")
        return 1

    docs = load_documents(pdf_paths, min_chars=args.min_chars)
    if not docs:
        logger.error("No usable documents could be loaded from the provided PDFs.")
        return 1

    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )
    if not chunks:
        logger.error("No usable chunks were created after cleaning.")
        return 1

    logger.info("Initializing embedding model: %s", args.embedding_model)
    base_embeddings = OpenAIEmbeddings(model=args.embedding_model)

    cache_dir = Path(args.cache_root) / args.embedding_model
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Embedding cache path: %s", cache_dir)

    store = LocalFileStore(str(cache_dir))
    embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=args.embedding_model,
    )
    logger.info("Embedding cache initialized.")

    index_root = Path(args.index_root)
    metadata_path = index_root / "build_metadata.json"
    fingerprint = fingerprint_files(pdf_paths)

    vectorstore = load_or_build_index(
        chunks=chunks,
        embeddings=embeddings,
        index_root=index_root,
        metadata_path=metadata_path,
        fingerprint=fingerprint,
        input_files=pdf_paths,
        force_rebuild=args.force_rebuild,
    )

    if args.query:
        run_query(vectorstore, args.query, args.top_k)

    logger.info("Vector DB build completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        sys.exit(1)