# 반도체 관련 pdf 파일 -> FAISS Vector DB 구축

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.storage import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFVectorDBBuilder:
    """Reference-style PDF -> chunk -> embedding cache -> FAISS index builder.

    Based on the uploaded tutorial code structure:
    - PDFPlumberLoader
    - RecursiveCharacterTextSplitter
    - OpenAIEmbeddings + LocalFileStore cache
    - FAISS save/load with doc hash
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        cache_root: str = ".cache/embeddings",
        index_root: str = ".cache/faiss_index",
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_root = Path(cache_root)
        self.index_root = Path(index_root)

    def _make_dataset_key(self, pdf_paths: List[str]) -> str:
        normalized = [str(Path(p).resolve()) for p in sorted(pdf_paths)]
        raw = "\n".join(normalized)
        dataset_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

        if len(normalized) == 1:
            stem = Path(normalized[0]).stem
            return f"{stem}_{dataset_hash}"
        return f"multi_pdf_{len(normalized)}_{dataset_hash}"

    def _validate_pdf_paths(self, pdf_paths: Iterable[str]) -> List[str]:
        valid_paths: List[str] = []
        invalid_paths: List[str] = []

        for path in pdf_paths:
            path_obj = Path(path)
            if not path_obj.exists() or not path_obj.is_file() or path_obj.suffix.lower() != ".pdf":
                invalid_paths.append(path)
                continue
            if not os.access(path_obj, os.R_OK):
                invalid_paths.append(path)
                continue
            valid_paths.append(str(path_obj))

        if invalid_paths:
            print("Skipped invalid PDF paths:")
            for item in invalid_paths:
                print(f"- {item}")

        if not valid_paths:
            raise ValueError("No valid readable PDF files were provided.")

        return valid_paths

    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        loaded_files = 0

        for pdf_path in pdf_paths:
            print(f"Loading PDF: {pdf_path}")
            loader = PDFPlumberLoader(pdf_path)
            loaded_docs = loader.load()
            if not loaded_docs:
                print(f"Warning: no content loaded from {pdf_path}")
                continue

            # metadata 보강
            for doc in loaded_docs:
                doc.metadata["file_name"] = Path(pdf_path).name
                doc.metadata["source_path"] = str(Path(pdf_path).resolve())
                # PDFPlumberLoader page is zero-based in metadata; normalize helper field
                if "page" in doc.metadata:
                    doc.metadata["page_number"] = int(doc.metadata["page"]) + 1

            docs.extend(loaded_docs)
            loaded_files += 1
            print(f"- loaded {len(loaded_docs)} pages")

        if not docs:
            raise ValueError("No documents could be loaded from the supplied PDFs.")

        print(f"Loaded {loaded_files} PDFs / {len(docs)} page documents")
        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = self.create_text_splitter()
        split_docs = splitter.split_documents(docs)

        # 청크 단위 metadata 보강
        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = idx
            doc.metadata["content_length"] = len(doc.page_content)

        return split_docs

    def create_embeddings(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        underlying_embeddings = OpenAIEmbeddings(model=self.embedding_model)
        store = LocalFileStore(str(cache_dir))
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            store,
            namespace=self.embedding_model,
            key_encoder="sha256",
        )

    def _documents_hash(self, docs: List[Document]) -> str:
        payload = "\n\n".join(
            f"{doc.metadata.get('source_path', '')}|{doc.metadata.get('page_number', '')}|{doc.page_content}"
            for doc in docs
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def build(self, pdf_paths: List[str], force_rebuild: bool = False) -> dict:
        pdf_paths = self._validate_pdf_paths(pdf_paths)
        dataset_key = self._make_dataset_key(pdf_paths)
        cache_dir = self.cache_root / dataset_key
        index_dir = self.index_root / dataset_key
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Dataset key: {dataset_key}")
        print(f"Embedding cache: {cache_dir}")
        print(f"FAISS index dir: {index_dir}")

        docs = self.load_documents(pdf_paths)
        split_docs = self.split_documents(docs)
        current_hash = self._documents_hash(split_docs)

        index_path = str(index_dir / "faiss_index")
        hash_file = index_dir / "doc_hash.txt"
        metadata_file = index_dir / "build_metadata.json"

        embeddings = self.create_embeddings(cache_dir)

        should_load_cache = (
            not force_rebuild
            and hash_file.exists()
            and Path(index_path + ".faiss").exists()
            and hash_file.read_text(encoding="utf-8").strip() == current_hash
        )

        if should_load_cache:
            print("Existing FAISS index matches current documents. Loading from disk...")
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("Creating new FAISS index...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(index_path)
            hash_file.write_text(current_hash, encoding="utf-8")
            metadata = {
                "dataset_key": dataset_key,
                "pdf_paths": [str(Path(p).resolve()) for p in pdf_paths],
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "num_page_docs": len(docs),
                "num_chunks": len(split_docs),
            }
            metadata_file.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("FAISS index saved.")

        return {
            "dataset_key": dataset_key,
            "cache_dir": str(cache_dir),
            "index_dir": str(index_dir),
            "num_pdfs": len(pdf_paths),
            "num_page_docs": len(docs),
            "num_chunks": len(split_docs),
            "vectorstore": vectorstore,
        }


def collect_pdf_paths(inputs: List[str], recursive: bool = True) -> List[str]:
    pdf_paths: List[str] = []

    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_paths.append(str(path))
        elif path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdf_paths.extend(str(p) for p in path.glob(pattern) if p.is_file())
        else:
            # glob 패턴 지원
            parent = Path(".")
            pdf_paths.extend(str(p) for p in parent.glob(item) if p.is_file() and p.suffix.lower() == ".pdf")

    # 순서 고정 + 중복 제거
    return sorted(set(pdf_paths))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build FAISS vector DB from PDF files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF files, directories, or glob patterns (e.g. './papers/*.pdf').",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--cache-root", default=".cache/embeddings")
    parser.add_argument("--index-root", default=".cache/faiss_index")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--query",
        default=None,
        help="Optional smoke-test query to run after index build.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    pdf_paths = collect_pdf_paths(args.inputs, recursive=not args.no_recursive)
    if not pdf_paths:
        raise ValueError("No PDF files found from the provided inputs.")

    builder = PDFVectorDBBuilder(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        cache_root=args.cache_root,
        index_root=args.index_root,
    )
    result = builder.build(pdf_paths, force_rebuild=args.force_rebuild)

    print("\nBuild complete")
    print(json.dumps({k: v for k, v in result.items() if k != "vectorstore"}, ensure_ascii=False, indent=2))

    if args.query:
        print(f"\nQuery smoke test: {args.query}")
        retriever = result["vectorstore"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.top_k},
        )
        docs = retriever.invoke(args.query)
        for i, doc in enumerate(docs, start=1):
            print(f"\n[{i}] {doc.metadata.get('file_name')} - page {doc.metadata.get('page_number')}")
            preview = doc.page_content[:300].replace("\n", " ")
            print(preview + ("..." if len(doc.page_content) > 300 else ""))


if __name__ == "__main__":
    main()
