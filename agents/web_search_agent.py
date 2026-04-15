# agents/web_search_agent.py
# 작성: 2반 이한결, 한채윤

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 도구 초기화 ──────────────────────────────────────────
tool = TavilySearch(max_results=30)

# ── 확증 편향 방지: 반대 관점 쿼리 3종 강제 생성 ──────────
QUERY_TEMPLATES = {
    "Samsung": [
        "Samsung HCB Hybrid Copper Bonding HBM4 2024 2025",           # 긍정
        "Samsung Hybrid Bonding HBM yield problem challenge",          # 부정
        "Samsung HBM4 delay technical difficulty",                     # 역관점
    ],
    "SK하이닉스": [
        "SK Hynix Hybrid Bonding HBM4 progress 2024 2025",            # 긍정
        "SK Hynix Hybrid Bonding yield problem challenge",             # 부정
        "SK Hynix MR-MUF Hybrid Bonding transition difficulty",        # 역관점
    ],
    "TSMC": [
        "TSMC SoIC Hybrid Bonding HBM base die 2024 2025",            # 긍정
        "TSMC SoIC Hybrid Bonding yield challenge limitation",         # 부정
        "TSMC Hybrid Bonding HBM integration difficulty",             # 역관점
    ],
}

# ── 웹 검색 수행 ──────────────────────────────────────────
def run_web_search(state: dict) -> dict:
    """
    Web Search Agent
    - 경쟁사 3사에 대해 긍정/부정/역관점 3종 쿼리 강제 실행
    - 확증 편향 방지: 동일 출처 비율 > 50% 시 재검색
    - 결과를 요약+메타 형태로 State에 저장 (토큰 절약)
    """
    print("\n[Web Search Agent] 웹 검색 시작...")

    all_results = []
    source_counter = {}  # 출처 다양성 체크용

    for competitor, queries in QUERY_TEMPLATES.items():
        print(f"  → {competitor} 검색 중...")

        for query in queries:
            try:
                raw = tool.invoke(query)
                results = raw.get("results", [])

                for r in results:
                    source = r.get("url", "")
                    domain = _extract_domain(source)

                    # 출처 카운트
                    source_counter[domain] = source_counter.get(domain, 0) + 1

                    all_results.append({
                        "summary": r.get("content", "")[:1000],  # 300자로 제한 (토큰 절약)
                        "source": source,
                        "date": r.get("published_date", "unknown"),
                        "keyword_matched": query,
                        "competitor": competitor,
                        "score": r.get("score", 0.0),
                    })

            except Exception as e:
                print(f"  ⚠️  검색 실패 ({query}): {e}")
                continue

    # ── 확증 편향 체크 ────────────────────────────────────
    all_results = _check_source_diversity(all_results, source_counter)

    # ── 결과 수 검증 ──────────────────────────────────────
    result_count = len(all_results)
    print(f"  → 총 {result_count}건 수집 완료")

    state["web_results"] = all_results
    state["web_search_count"] = result_count

    return state


# ── 확증 편향 방지: 동일 출처 50% 초과 시 경고 ───────────
def _check_source_diversity(results: list, source_counter: dict) -> list:
    total = len(results)
    if total == 0:
        return results

    for domain, count in source_counter.items():
        ratio = count / total
        if ratio > 0.5:
            print(f"  ⚠️  확증 편향 경고: {domain} 출처 비율 {ratio:.0%} > 50%")
            print(f"      → 해당 출처 결과 일부 제거")
            # 동일 출처 최대 3건으로 제한
            filtered = []
            domain_count = 0
            for r in results:
                if _extract_domain(r["source"]) == domain:
                    if domain_count < 3:
                        filtered.append(r)
                        domain_count += 1
                else:
                    filtered.append(r)
            results = filtered

    return results


# ── 도메인 추출 유틸 ──────────────────────────────────────
def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except:
        return url


# ── 단독 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "query": "HBM4 Hybrid Bonding 경쟁사 분석",
        "web_results": [],
        "web_search_count": 0,
    }

    result = run_web_search(test_state)

    print("\n=== 수집 결과 샘플 ===")
    for i, r in enumerate(result["web_results"][:3], 1):
        print(f"\n[{i}] {r['competitor']} | {r['keyword_matched']}")
        print(f"    출처: {r['source']}")
        print(f"    요약: {r['summary'][:100]}...")