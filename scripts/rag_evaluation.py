# scripts/rag_evaluation.py
# 작성: 2반 이한결, 한채윤
# 설명: HBM4 Hybrid Bonding 논문 데이터 기반 RAG Retriever 평가
#       평가 지표: Hit Rate@K, MRR

import os
import glob
import random
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
)

warnings.filterwarnings("ignore")
load_dotenv()

# ── 한글 폰트 설정 (macOS) ────────────────────────────────
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 설정값 ────────────────────────────────────────────────
PAPERS_DIR = "./data/papers"
BASE_CHUNK_SIZE = 800
BASE_CHUNK_OVERLAP = 150
N_EVAL_SAMPLES = 20
MIN_CHUNK_LENGTH = 100
K_VALUES = [1, 3, 5]
RANDOM_SEED = 42


# ════════════════════════════════════════════════════════
# 1. 문서 로딩 및 청킹
# ════════════════════════════════════════════════════════
def load_documents(papers_dir: str) -> list:
    pdf_paths = sorted(glob.glob(f"{papers_dir}/*.pdf"))
    print(f"[1] PDF 로딩 중... ({len(pdf_paths)}편)")

    raw_docs = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        raw_docs.extend(docs)
        print(f"  → {os.path.basename(path)} ({len(docs)}페이지)")

    print(f"  총 페이지 수: {len(raw_docs)}\n")
    return raw_docs


def chunk_documents(raw_docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASE_CHUNK_SIZE,
        chunk_overlap=BASE_CHUNK_OVERLAP,
    )
    base_chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(base_chunks):
        chunk.metadata["chunk_id"] = i

    print(f"[2] 청킹 완료: {len(base_chunks)}개 청크")
    print(f"  평균 길이: {sum(len(c.page_content) for c in base_chunks) / len(base_chunks):.0f}자\n")
    return base_chunks


# ════════════════════════════════════════════════════════
# 2. QA 데이터셋 자동 생성 (LLM)
# ════════════════════════════════════════════════════════
def generate_eval_dataset(base_chunks: list, llm) -> list:
    random.seed(RANDOM_SEED)

    eligible = [c for c in base_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
    sampled = random.sample(eligible, min(N_EVAL_SAMPLES, len(eligible)))

    print(f"[3] QA 자동 생성 중... ({len(sampled)}개)")

    prompt = PromptTemplate.from_template("""
다음은 HBM4 Hybrid Bonding 관련 논문의 일부입니다.

<content>
{content}
</content>

위 내용을 읽고, 이 내용으로 답할 수 있는 구체적인 질문 1개를 생성하세요.

조건:
- 질문은 위 내용에서 답을 찾을 수 있어야 합니다
- 기술적인 내용을 포함한 구체적인 질문이어야 합니다
- 반드시 아래 JSON 형식으로만 응답하세요

{{"question": "질문 내용"}}
""")

    chain = prompt | llm | JsonOutputParser()
    eval_dataset = []

    for i, chunk in enumerate(sampled):
        try:
            result = chain.invoke({"content": chunk.page_content})
            eval_dataset.append({
                "question": result["question"],
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_content": chunk.page_content,
            })
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(sampled)} 완료")
        except Exception as e:
            print(f"  청크 {chunk.metadata['chunk_id']} 생성 실패: {e}")

    print(f"  총 {len(eval_dataset)}개 질문 생성 완료\n")

    # 샘플 출력
    for i, row in enumerate(eval_dataset[:3]):
        print(f"  [Q{i+1}] {row['question']}")

    print()
    return eval_dataset


# ════════════════════════════════════════════════════════
# 3. 평가 함수
# ════════════════════════════════════════════════════════
def is_relevant(retrieved_doc: Document, ground_truth_content: str, threshold: float = 0.3) -> bool:
    retrieved = retrieved_doc.page_content
    gt = ground_truth_content

    if gt[:100] in retrieved or retrieved[:100] in gt:
        return True

    window_size, step = 30, 15
    windows = [gt[i:i+window_size] for i in range(0, len(gt) - window_size + 1, step)]

    if not windows:
        return gt in retrieved

    matched = sum(1 for w in windows if w in retrieved)
    return (matched / len(windows)) >= threshold


def hit_rate_at_k(retriever, eval_data: list, k: int, threshold: float = 0.3) -> float:
    hits = 0
    for item in eval_data:
        try:
            results = retriever.invoke(item["question"])[:k]
            if any(is_relevant(doc, item["chunk_content"], threshold) for doc in results):
                hits += 1
        except Exception:
            pass
    return hits / len(eval_data)


def mrr_score(retriever, eval_data: list, k: int = 10, threshold: float = 0.3) -> float:
    reciprocal_ranks = []
    for item in eval_data:
        try:
            results = retriever.invoke(item["question"])[:k]
            rr = 0.0
            for rank, doc in enumerate(results, start=1):
                if is_relevant(doc, item["chunk_content"], threshold):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        except Exception:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retriever(name: str, retriever, eval_data: list, k_values: list) -> dict:
    result = {"Retriever": name}
    print(f"  평가 중: {name}")

    for k in k_values:
        result[f"Hit@{k}"] = round(hit_rate_at_k(retriever, eval_data, k=k), 4)

    result["MRR"] = round(mrr_score(retriever, eval_data, k=max(k_values)), 4)

    hit_str = ", ".join(f"Hit@{k}={result[f'Hit@{k}']:.3f}" for k in k_values)
    print(f"    → {hit_str}, MRR={result['MRR']:.3f}")
    return result


# ════════════════════════════════════════════════════════
# 4. Retriever 구성 및 평가
# ════════════════════════════════════════════════════════
def build_and_evaluate(base_chunks, raw_docs, eval_dataset, embeddings, llm):
    all_results = []

    print("[5] Retriever 평가 시작...\n")

    # 공통 FAISS
    faiss_db = FAISS.from_documents(base_chunks, embeddings)
    print(f"  FAISS 인덱스 구성 완료 ({len(base_chunks)}개 청크)\n")

    # 1. FAISS Similarity
    r1 = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(K_VALUES)}
    )
    all_results.append(evaluate_retriever("FAISS (Similarity)", r1, eval_dataset, K_VALUES))

    # 2. FAISS MMR
    r2 = faiss_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(K_VALUES), "fetch_k": 20, "lambda_mult": 0.5}
    )
    all_results.append(evaluate_retriever("FAISS (MMR)", r2, eval_dataset, K_VALUES))

    # 3. BM25
    r3 = BM25Retriever.from_documents(base_chunks, k=max(K_VALUES))
    all_results.append(evaluate_retriever("BM25 (기본)", r3, eval_dataset, K_VALUES))

    # 4. Ensemble (BM25 + FAISS) ← 현재 선정값
    bm25_ens = BM25Retriever.from_documents(base_chunks, k=max(K_VALUES))
    faiss_ens = faiss_db.as_retriever(search_kwargs={"k": max(K_VALUES)})
    r4 = EnsembleRetriever(retrievers=[bm25_ens, faiss_ens], weights=[0.5, 0.5])
    all_results.append(evaluate_retriever("Ensemble (BM25+FAISS)", r4, eval_dataset, K_VALUES))

    # 5. MultiQuery
    r5 = MultiQueryRetriever.from_llm(
        retriever=faiss_db.as_retriever(search_kwargs={"k": max(K_VALUES)}),
        llm=llm
    )
    all_results.append(evaluate_retriever("MultiQuery", r5, eval_dataset, K_VALUES))

    # Kiwi BM25 (선택사항 — kiwipiepy 설치 시)
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()

        def kiwi_tokenize(text: str) -> list:
            result = kiwi.tokenize(text)
            return [t.form for t in result if t.tag in ("NNG", "NNP", "VV", "VA", "SL")]

        r6 = BM25Retriever.from_documents(base_chunks, preprocess_func=kiwi_tokenize, k=max(K_VALUES))
        all_results.append(evaluate_retriever("Kiwi BM25", r6, eval_dataset, K_VALUES))

        kiwi_ens = BM25Retriever.from_documents(base_chunks, preprocess_func=kiwi_tokenize, k=max(K_VALUES))
        faiss_kiwi = faiss_db.as_retriever(search_kwargs={"k": max(K_VALUES)})
        r7 = EnsembleRetriever(retrievers=[kiwi_ens, faiss_kiwi], weights=[0.5, 0.5])
        all_results.append(evaluate_retriever("Kiwi Ensemble", r7, eval_dataset, K_VALUES))

    except ImportError:
        print("  ⚠️  kiwipiepy 미설치 → Kiwi 관련 평가 건너뜀")
        print("     설치: pip install kiwipiepy\n")

    return all_results


# ════════════════════════════════════════════════════════
# 5. 결과 출력 및 시각화
# ════════════════════════════════════════════════════════
def print_results(all_results: list):
    results_df = pd.DataFrame(all_results).set_index("Retriever")
    results_df = results_df.sort_values("MRR", ascending=False)

    print("\n" + "="*55)
    print("최종 평가 결과")
    print("="*55)
    print(results_df.to_string(float_format="{:.4f}".format))

    max_k = max(K_VALUES)
    print(f"\n  최고 MRR:    {results_df['MRR'].idxmax()} ({results_df['MRR'].max():.4f})")
    print(f"  최고 Hit@1:  {results_df['Hit@1'].idxmax()} ({results_df['Hit@1'].max():.4f})")
    print(f"  최고 Hit@{max_k}:  {results_df[f'Hit@{max_k}'].idxmax()} ({results_df[f'Hit@{max_k}'].max():.4f})")

    return results_df


def save_chart(results_df: pd.DataFrame, output_path: str = "outputs/rag_eval_result.png"):
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [f"Hit@{k}" for k in sorted(K_VALUES)] + ["MRR"]
    x = range(len(results_df))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2) * width + width / 2
        bars = ax.bar([xi + offset for xi in x], results_df[metric], width, label=metric, color=colors[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("HBM4 RAG Retriever 평가 결과 (Hit Rate & MRR)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n  차트 저장: {output_path}")
    plt.show()


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("HBM4 RAG Retriever 평가")
    print("=" * 55 + "\n")

    # 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. 문서 로딩
    raw_docs = load_documents(PAPERS_DIR)

    # 2. 청킹
    base_chunks = chunk_documents(raw_docs)

    # 3. QA 생성
    eval_dataset = generate_eval_dataset(base_chunks, llm)

    # 4. 평가
    all_results = build_and_evaluate(base_chunks, raw_docs, eval_dataset, embeddings, llm)

    # 5. 결과 출력
    results_df = print_results(all_results)

    # 6. 차트 저장
    save_chart(results_df)

    # 7. CSV 저장
    csv_path = "outputs/rag_eval_result.csv"
    results_df.to_csv(csv_path)
    print(f"  CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()