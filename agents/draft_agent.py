# agents/draft_agent.py
# 작성: 2반 이한결, 한채윤

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── LLM 초기화 ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=4000)

# ── 경로 설정 ─────────────────────────────────────────────
PROMPT_PATH = Path("prompts/draft_prompt.txt")
TEMPLATE_PATH = Path("data/report_template.md")  # ← 템플릿 경로

def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def _load_template() -> str:
    if not TEMPLATE_PATH.exists():
        print(f"  ⚠️  템플릿 파일 없음: {TEMPLATE_PATH}")
        return ""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ── Draft Agent 실행 ──────────────────────────────────────
def run_draft(state: dict) -> dict:
    """
    Draft Agent
    - report_template.md 구조를 기반으로 보고서 초안 작성
    - Analysis Agent 결과를 템플릿의 빈칸에 채워넣는 방식
    - Self-Reflection으로 SC 체크
    """
    print("\n[Draft Agent] 보고서 초안 작성 시작...")

    analysis_results = state.get("analysis_results", {})
    limitation_summary = state.get("limitation_summary", "")
    retry_draft = state.get("retry_draft", 0)
    reflection_feedback = state.get("reflection_feedback", "")

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
템플릿의 빈칸([   ])을 Analysis 결과로 채우고,
내용이 없는 항목은 수집된 데이터 기반으로 작성하세요.

{template}
"""

    user_message = f"""
다음 분석 결과를 바탕으로 보고서 초안을 작성하세요.

[Analysis 결과]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[정보 가용성 한계]
{limitation_summary}

{template_section}

{feedback_section}

작성 규칙:
1. 템플릿 구조(목차, 섹션)를 그대로 유지할 것
2. TRL 추정값은 Analysis 결과에서 그대로 가져올 것
3. 경쟁사별 추정 근거, R&D 전략, 최신 동향을 구체적으로 채울 것
4. 3.4 경쟁사 비교 종합 표를 반드시 완성할 것
5. REFERENCE 섹션에 실제 출처 URL을 포함할 것
6. 보고서 작성 후 Self-Reflection 체크리스트를 포함할 것
7. 각 섹션은 최소 3~5문장 이상 작성할 것 
8. 경쟁사별 분석은 수집된 증거(evidence)와 간접 지표를 모두 서술할 것 
9. SUMMARY는 보고서 전체 내용을 압축하여 임원이 바로 판단할 수 있도록 작성할 것 
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

    return state


# ── Draft / Reflection 분리 파싱 ─────────────────────────
def _parse_draft_and_reflection(raw: str) -> tuple:
    reflection = {
        "sc1": "NO", "sc2": "NO", "sc3": "NO",
        "sc4": "NO", "sc5": "NO", "sc6": "NO",
        "all_passed": "NO",
        "failed_items": [],
    }

    if "self_reflection:" not in raw:
        return raw, reflection

    parts = raw.split("self_reflection:")
    draft = parts[0].strip()
    reflection_raw = parts[1].strip() if len(parts) > 1 else ""

    failed = []
    for sc in ["sc1", "sc2", "sc3", "sc4", "sc5", "sc6"]:
        if f"{sc}: yes" in reflection_raw.lower():
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
    print(test_state['draft'])
    print(f"\n=== SC 통과 여부: {test_state['draft_passed']} ===")
    print(f"미충족 항목: {test_state['reflection_feedback']}")