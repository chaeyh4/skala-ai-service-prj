# agents/analysis_agent.py
# 작성: 2반 이한결, 한채윤

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── LLM 초기화 ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=3000)

# ── 프롬프트 로드 ─────────────────────────────────────────
PROMPT_PATH = Path("prompts/analysis_prompt.txt")
TRL_PATH = Path("data/TRL.md")

def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def _load_trl_guideline() -> str:
    with open(TRL_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ── Analysis Agent 실행 ───────────────────────────────────
def run_analysis(state: dict) -> dict:
    """
    Analysis Agent
    - RAG + Web 결과 품질 평가 (criteria_scores)
    - 경쟁사별 TRL 추정
    - 위협 수준 평가
    - 정보 가용성 한계 명시
    """
    print("\n[Analysis Agent] 분석 시작...")

    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])

    # ── 입력 데이터 요약 ──────────────────────────────────
    rag_summary = json.dumps(rag_results[:100], ensure_ascii=False, indent=2)
    web_summary = json.dumps(web_results[:150], ensure_ascii=False, indent=2)
    trl_guideline = _load_trl_guideline()
    system_prompt = _load_prompt()

    user_message = f"""
다음 데이터를 분석하여 JSON 형태로만 반환하세요.

[RAG 검색 결과]
{rag_summary}

[웹 검색 결과]
{web_summary}

[TRL 기준]
{trl_guideline}
"""

    # ── LLM 호출 ──────────────────────────────────────────
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])

        raw = response.content.strip()

        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        analysis_output = json.loads(raw)
        print("  → 분석 완료")

    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON 파싱 실패: {e}")
        analysis_output = _fallback_output()
    except Exception as e:
        print(f"  ⚠️  LLM 호출 실패: {e}")
        analysis_output = _fallback_output()

    # ── State 업데이트 ────────────────────────────────────
    state["sc_scores"] = analysis_output.get("criteria_scores", {})
    state["analysis_results"] = analysis_output.get("analysis_results", {})
    state["source_count"] = analysis_output.get("source_count", 0)
    state["limitation_summary"] = analysis_output.get("limitation_summary", "")

    # ── Supervisor 보고 조건 체크 ─────────────────────────
    scores = state["sc_scores"]
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    source_count = state["source_count"]

    state["analysis_passed"] = (
        avg_score >= 0.7 and source_count >= 7
    )

    print(f"  → criteria_scores 평균: {avg_score:.2f}")
    print(f"  → 출처 수: {source_count}")
    print(f"  → 통과 여부: {state['analysis_passed']}")

    return state


# ── Fallback 출력 ─────────────────────────────────────────
def _fallback_output() -> dict:
    """LLM 실패 시 기본 구조 반환"""
    return {
        "criteria_scores": {
            "sc1_competitors": 0.0,
            "sc2_trl": 0.0,
            "sc3_evidence": 0.0,
        },
        "source_count": 0,
        "analysis_results": {
            "SK하이닉스": {
                "trl_level": 0,
                "confidence": "낮음",
                "evidence": [],
                "indirect_indicators": [],
                "information_gap": "분석 실패",
                "assessment_note": "LLM 호출 실패로 분석 불가",
                "threat_level": 0,
            },
            "Samsung": {
                "trl_level": 0,
                "confidence": "낮음",
                "evidence": [],
                "indirect_indicators": [],
                "information_gap": "분석 실패",
                "assessment_note": "LLM 호출 실패로 분석 불가",
                "threat_level": 0,
            },
            "TSMC": {
                "trl_level": 0,
                "confidence": "낮음",
                "evidence": [],
                "indirect_indicators": [],
                "information_gap": "분석 실패",
                "assessment_note": "LLM 호출 실패로 분석 불가",
                "threat_level": 0,
            },
        },
        "limitation_summary": "분석 실패로 한계 명시 불가",
    }


# ── 단독 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    from web_search_agent import run_web_search
    from rag_agent import get_rag_agent
    
    rag_agent = get_rag_agent()
    rag_output = rag_agent.run("HBM4 hybrid bonding yield Samsung TSMC")

    test_state = {
        "query": "HBM4 Hybrid Bonding 경쟁사 분석",
        "rag_results": rag_output["rag_results"],  # 실제 RAG 결과
        "web_results": [],
    }

    test_state = run_web_search(test_state)
    test_state = run_analysis(test_state)

    print("\n=== Analysis 결과 ===")
    print(json.dumps(test_state["analysis_results"], ensure_ascii=False, indent=2))
    print(f"\n요약: {test_state}")