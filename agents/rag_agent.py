from __future__ import annotations

import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


logger = logging.getLogger(__name__)


@dataclass
class RAGAgentConfig:
    """Configuration for the RAG Agent."""
    index_root: str = ".cache/faiss_index"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 20
    min_results: int = 5


class RAGAgent:
    """
    RAG Agent for querying a pre-built FAISS vector DB.

    Design alignment:
    - Role: Search only on a pre-built DB
    - Input: search keyword(s)
    - Output: list[dict] with summary + source + trl_signal + chunk_id
    - Supervisor report condition: retry if result count < 5
    """

    def __init__(self, config: Optional[RAGAgentConfig] = None) -> None:
        load_dotenv()
        self.config = config or RAGAgentConfig()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set.")

        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.vectorstore = FAISS.load_local(
            self.config.index_root,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("RAGAgent initialized with index at %s", self.config.index_root)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search the vector DB and return design-spec compliant results.

        Returns:
            list[dict]:
            [
              {
                "summary": "...",
                "source": "data/papers/file.pdf",
                "trl_signal": "...",
                "chunk_id": "file.pdf:p3:c12"
              }
            ]
        """
        k = top_k or self.config.top_k
        docs = self.vectorstore.similarity_search(query, k=k)

        results: List[Dict[str, Any]] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            filename = doc.metadata.get("filename", Path(source).name)
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            text = self._clean_for_summary(doc.page_content)

            result = {
                "summary": self._make_summary(text),
                "source": source,
                "trl_signal": self._extract_trl_signal(text),
                "chunk_id": f"{filename}:p{page}:c{chunk_id}",
            }
            results.append(result)

        return results

    def should_retry(self, results: List[Dict[str, Any]]) -> bool:
        """Supervisor rule: retry if result count < 5."""
        return len(results) < self.config.min_results

    def run(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Convenience wrapper for supervisor integration.

        Returns:
            {
              "rag_results": list[dict],
              "result_count": int,
              "retry_needed": bool
            }
        """
        results = self.search(query=query, top_k=top_k)
        return {
            "rag_results": results,
            "result_count": len(results),
            "retry_needed": self.should_retry(results),
        }

    @staticmethod
    def _clean_for_summary(text: str) -> str:
        text = text or ""
        text = re.sub(r"\(cid:\d+\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _make_summary(text: str, max_chars: int = 1000) -> str:
        """
        Create a compact 2-3 sentence style summary from a chunk.
        For chunked paper text, a clipped summary is usually more stable than LLM summarization.
        """
        if not text:
            return ""

        # Prefer sentence-ish splitting, then fallback to clipping
        parts = re.split(r"(?<=[.!?])\s+", text)
        summary = " ".join(parts[:2]).strip()
        if len(summary) < 80:
            summary = text[:max_chars].strip()

        return summary[:max_chars].rstrip()

    @staticmethod
    def _extract_trl_signal(text: str) -> str:
        """
        Heuristic TRL signal extraction aligned with the design doc.
        """
        lowered = text.lower()

        rules = [
            (["concept", "idea", "principle", "model"], "이론/개념 중심 → TRL 2~3 신호"),
            (["simulation", "modeled", "laboratory", "experiment"], "실험실 수준 검증 → TRL 3~4 신호"),
            (["prototype", "demonstration", "demo"], "시작품/데모 → TRL 4~5 신호"),
            (["yield", "throughput", "performance", "electrical test"], "성능 수치 공개 → TRL 5~6 신호"),
            (["reliability", "qualification", "customer sample"], "신뢰성/고객 평가 → TRL 6~7 신호"),
            (["production", "mass production", "volume manufacturing", "hvm"], "양산 준비/양산 → TRL 8~9 신호"),
        ]

        for keywords, signal in rules:
            if any(keyword in lowered for keyword in keywords):
                return signal

        return "직접 수치 부족 → 간접 근거 기반 TRL 추정 필요"


def get_rag_agent(
    index_root: str = ".cache/faiss_index",
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 20,
    min_results: int = 5,
) -> RAGAgent:
    """Factory function for app/supervisor integration."""
    config = RAGAgentConfig(
        index_root=index_root,
        embedding_model=embedding_model,
        top_k=top_k,
        min_results=min_results,
    )
    return RAGAgent(config=config)


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG agent search against a pre-built FAISS DB.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index-root", default=".cache/faiss_index")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    agent = get_rag_agent(
        index_root=args.index_root,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    output = agent.run(args.query)
    print(json.dumps(output, indent=2, ensure_ascii=False))
