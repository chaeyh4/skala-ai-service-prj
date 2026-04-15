# agents/draft_agent.py
# 작성: 2반 이한결, 한채윤

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── LLM 초기화 ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=7000)

# ── 경로 설정 ─────────────────────────────────────────────
PROMPT_PATH = Path("prompts/draft_prompt.txt")
TEMPLATE_PATH = Path("data/report_template.md")


def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _load_template() -> str:
    if not TEMPLATE_PATH.exists():
        print(f"  ⚠️  템플릿 파일 없음: {TEMPLATE_PATH}")
        return ""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ── 출처 풀 구성 유틸 ─────────────────────────────────────
def _normalize_source(source: str) -> str:
    if not source:
        return ""
    return source.strip()


def _extract_domain(source: str) -> str:
    try:
        return urlparse(source).netloc or "unknown"
    except Exception:
        return "unknown"


def _dedup_source_pool(
    rag_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    max_sources: int = 15,
) -> List[Dict[str, Any]]:
    """
    Draft Agent에 넘길 참고 출처 풀 생성
    - 중복 URL 제거
    - 너무 긴 raw input 방지 위해 상위 N개만 유지
    - RAG / Web 출처를 균형 있게 섞음
    """
    seen = set()
    pool: List[Dict[str, Any]] = []

    # RAG 먼저
    for item in rag_results:
        source = _normalize_source(item.get("source", ""))
        if not source or source in seen:
            continue
        seen.add(source)
        pool.append({
            "source": source,
            "source_type": "rag",
            "domain": _extract_domain(source),
            "summary": item.get("summary", "")[:300],
            "trl_signal": item.get("trl_signal", ""),
            "chunk_id": item.get("chunk_id", ""),
        })
        if len(pool) >= max_sources:
            return pool

    # Web 추가
    for item in web_results:
        source = _normalize_source(item.get("source", ""))
        if not source or source in seen:
            continue
        seen.add(source)
        pool.append({
            "source": source,
            "source_type": "web",
            "domain": _extract_domain(source),
            "summary": item.get("summary", "")[:300],
            "date": item.get("date", "unknown"),
            "competitor": item.get("competitor", ""),
            "keyword_matched": item.get("keyword_matched", ""),
        })
        if len(pool) >= max_sources:
            break

    return pool


def _collect_reference_candidates(
    analysis_results: Dict[str, Any],
    draft_source_pool: List[Dict[str, Any]],
    max_candidates: int = 20,
) -> List[str]:
    """
    Analysis evidence + source pool에서 reference 후보 URL 수집
    """
    refs: List[str] = []
    seen = set()

    # analysis_results 내 evidence / indirect_indicators에서 추출
    for company_data in analysis_results.values():
        if not isinstance(company_data, dict):
            continue

        for field in ["evidence", "indirect_indicators"]:
            values = company_data.get(field, [])
            if not isinstance(values, list):
                continue

            for v in values:
                text = str(v).strip()
                if not text:
                    continue
                # 아주 단순한 URL 추출 보조
                for token in text.split():
                    if token.startswith("http://") or token.startswith("https://"):
                        if token not in seen:
                            seen.add(token)
                            refs.append(token)
                            if len(refs) >= max_candidates:
                                return refs

    # source pool의 source 추가
    for item in draft_source_pool:
        source = item.get("source", "").strip()
        if source and source not in seen:
            seen.add(source)
            refs.append(source)
            if len(refs) >= max_candidates:
                break

    return refs


# ── Draft Agent 실행 ──────────────────────────────────────
def run_draft(state: dict) -> dict:
    """
    Draft Agent
    - report_template.md 구조를 기반으로 보고서 초안 작성
    - Analysis Agent 결과를 템플릿의 빈칸에 채워넣는 방식
    - 원본 출처 풀(draft_source_pool)을 함께 전달하여 REFERENCE 품질 강화
    - Self-Reflection으로 SC 체크
    """
    print("\n[Draft Agent] 보고서 초안 작성 시작...")

    analysis_results = state.get("analysis_results", {})
    limitation_summary = state.get("limitation_summary", "")
    retry_draft = state.get("retry_draft", 0)
    reflection_feedback = state.get("reflection_feedback", "")

    # ── 원본 데이터 주입 ──────────────────────────────────────
    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])

    # 상위 결과만 선별 (토큰 절약)
    rag_top = rag_results[:8]
    web_top = sorted(web_results, key=lambda x: x.get("score", 0), reverse=True)[:20]

    system_prompt = _load_prompt()
    template = _load_template()

    # ── 수정 요청 피드백 ──────────────────────────────────
    feedback_section = ""
    if reflection_feedback:
        feedback_section = f"""
[이전 검토 피드백 — 반드시 반영할 것]
{reflection_feedback}
"""

    # ── 템플릿 섹션 ───────────────────────────────────────
    template_section = ""
    if template:
        template_section = f"""
[보고서 템플릿 — 이 구조를 반드시 따를 것]
{template}
"""

    paper_dir = Path("data/papers")
    paper_list = "\n".join([
        f"- {f.stem.replace('_', ' ')}"
        for f in sorted(paper_dir.glob("*.pdf"))
    ])

    user_message = f"""
다음 데이터를 바탕으로 보고서 초안을 작성하세요.

[Analysis 결과 — TRL 판단 근거]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[정보 가용성 한계]
{limitation_summary}

[RAG 논문 원본 데이터 — 기술 내용 작성 시 직접 활용]
{json.dumps(rag_top, ensure_ascii=False, indent=2)}

[웹 검색 원본 데이터 — 경쟁사 동향 작성 시 직접 활용]
{json.dumps(web_top, ensure_ascii=False, indent=2)}

{template_section}

{source_pool_section}

{reference_candidate_section}

{feedback_section}

작성 규칙:
1. 템플릿 구조(목차, 섹션)를 그대로 유지할 것
2. TRL 추정값은 Analysis 결과에서 그대로 가져올 것
3. 경쟁사별 분석은 RAG/웹 원본 데이터에서 직접 근거를 인용할 것
   예: "(출처: 논문명)" 또는 "(출처: URL)"
4. 3.4 경쟁사 비교 종합 표를 반드시 완성할 것
5. REFERENCE 섹션에 실제 출처 URL/논문명 포함할 것
6. REFERENCE 섹션에 위 전체 논문 목록 중 관련 논문을 최대한 많이 포함할 것
7. 보고서 작성 후 self_reflection: 섹션 반드시 포함할 것
8. 각 섹션은 최소 5문장 이상 구체적으로 서술할 것
9. 단순 목록 나열 금지 — 문장으로 풀어서 작성할 것
10. SUMMARY는 임원이 읽고 바로 판단할 수 있는 수준으로 작성할 것
"""

    # ── LLM 호출 ──────────────────────────────────────────
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        raw = response.content.strip()
        print("  → 초안 작성 완료")

    except Exception as e:
        print(f"  ⚠️  LLM 호출 실패: {e}")
        state["draft"] = ""
        state["draft_passed"] = False
        return state

    # ── Self-Reflection 파싱 ──────────────────────────────
    draft, reflection = _parse_draft_and_reflection(raw)
    all_passed = reflection.get("all_passed", "NO") == "YES"
    failed_items = reflection.get("failed_items", [])

    print(f"  → Self-Reflection: {'통과 ✅' if all_passed else '미충족 항목 존재 ⚠️'}")
    if not all_passed:
        print(f"  → 미충족 항목: {failed_items}")

    # ── State 업데이트 ────────────────────────────────────
    state["draft"] = draft
    state["reflection_feedback"] = (
        f"미충족 항목: {failed_items}" if not all_passed else ""
    )
    state["draft_passed"] = all_passed
    state["retry_draft"] = retry_draft
    state["draft_reference_candidates"] = reference_candidates

    return state


# ── Draft / Reflection 분리 파싱 ─────────────────────────
def _parse_draft_and_reflection(raw: str) -> Tuple[str, Dict[str, Any]]:
    reflection = {
        "sc1": "NO",
        "sc2": "NO",
        "sc3": "NO",
        "sc4": "NO",
        "sc5": "NO",
        "sc6": "NO",
        "sc7": "NO",
        "sc8": "NO",
        "all_passed": "NO",
        "failed_items": [],
    }

    if "self_reflection:" not in raw:
        return raw, reflection

    parts = raw.split("self_reflection:")
    draft = parts[0].strip()
    reflection_raw = parts[1].strip() if len(parts) > 1 else ""
    reflection_lower = reflection_raw.lower()

    failed = []
    check_items = ["sc1", "sc2", "sc3", "sc4", "sc5", "sc6", "sc7", "sc8"]

    for sc in check_items:
        if f"{sc}: yes" in reflection_lower:
            reflection[sc] = "YES"
        else:
            reflection[sc] = "NO"
            failed.append(sc.upper())

    reflection["failed_items"] = failed
    reflection["all_passed"] = "YES" if not failed else "NO"

    return draft, reflection


# ── 단독 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    from web_search_agent import run_web_search
    from rag_agent import get_rag_agent
    from analysis_agent import run_analysis

    rag_agent = get_rag_agent()
    rag_output = rag_agent.run("HBM4 hybrid bonding yield Samsung TSMC SK hynix")

    test_state = {
        "query": "HBM4 Hybrid Bonding 경쟁사 분석",
        "rag_results": rag_output["rag_results"],
        "web_results": [],
        "retry_draft": 0,
        "reflection_feedback": "",
    }

    test_state = run_web_search(test_state)
    test_state = run_analysis(test_state)
    test_state = run_draft(test_state)

    print("\n=== 보고서 초안 ===")
    print(test_state["draft"][:3000])
    print(f"\n=== SC 통과 여부: {test_state['draft_passed']} ===")
    print(f"미충족 항목: {test_state['reflection_feedback']}")
    print(f"REFERENCE 후보 수: {len(test_state.get('draft_reference_candidates', []))}")