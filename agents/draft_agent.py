# agents/draft_agent.py
# 작성: 2반 이한결, 한채윤

import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=6000)

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


def _parse_draft_and_reflection(raw: str) -> tuple:
    """보고서 본문과 self_reflection 섹션 분리"""
    reflection = {
        "sc1": "NO", "sc2": "NO", "sc3": "NO",
        "sc4": "NO", "sc5": "NO", "sc6": "NO",
        "all_passed": "NO",
        "failed_items": [],
    }

    separators = [
        "self_reflection:",
        "self-reflection:",
        "```yaml",
        "---\nself_reflection",
    ]

    draft = raw
    reflection_raw = ""

    for sep in separators:
        if sep.lower() in raw.lower():
            idx = raw.lower().find(sep.lower())
            draft = raw[:idx].strip()
            reflection_raw = raw[idx:].strip()
            break

    if not reflection_raw:
        return draft, reflection

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


def run_draft(state: dict) -> dict:
    print("\n[Draft Agent] 보고서 초안 작성 시작...")

    # 1. state에서 분석 결과뿐만 아니라 원본 검색 결과도 가져옵니다.
    analysis_results = state.get("analysis_results", {})
    limitation_summary = state.get("limitation_summary", "")
    rag_results = state.get("rag_results", [])  # RAG 원본 데이터 추가
    web_results = state.get("web_results", [])  # 웹 검색 원본 데이터 추가
    
    retry_draft = state.get("retry_draft", 0)
    reflection_feedback = state.get("reflection_feedback", "")

    system_prompt = _load_prompt()
    template = _load_template()

    feedback_section = ""
    if reflection_feedback:
        feedback_section = f"""
[이전 검토 피드백 — 반드시 반영할 것]
{reflection_feedback}
"""

    template_section = ""
    if template:
        template_section = f"""
[보고서 템플릿 — 이 구조를 반드시 따를 것]
템플릿의 빈칸([   ])을 Analysis 결과로 채우고,
내용이 없는 항목은 수집된 데이터 기반으로 작성하세요.

{template}
"""

    # 2. user_message에 원본 데이터를 주입합니다.
    user_message = f"""
다음 분석 결과와 수집된 원본 데이터를 바탕으로 보고서 초안을 전문적이고 상세하게 작성하세요.

[1. Analysis 결과 (요약)]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[2. RAG 검색 원본 데이터 (상세 근거 및 기술 데이터용)]
{json.dumps(rag_results, ensure_ascii=False, indent=2)}

[3. 웹 검색 원본 데이터 (최신 동향 및 시장 반응용)]
{json.dumps(web_results, ensure_ascii=False, indent=2)}

[4. 정보 가용성 한계]
{limitation_summary}

{template_section}

{feedback_section}

작성 규칙:
1. 템플릿 구조(목차, 섹션)를 그대로 유지할 것
2. TRL 추정값은 Analysis 결과에서 가져오되, 그에 대한 상세 설명은 원본 데이터(RAG/Web)를 참조하여 풍성하게 작성할 것
3. 경쟁사별 추정 근거, R&D 전략, 최신 동향에 수치나 구체적인 기술 명칭을 포함할 것
4. REFERENCE 섹션에 실제 출처 URL과 논문명을 누락 없이 포함할 것
5. 각 섹션은 최소 5문장 이상 작성하며, 단순 나열이 아닌 분석적인 문장으로 구성할 것
6. SUMMARY는 보고서 전체 내용을 압축하여 임원이 바로 판단할 수 있도록 작성할 것
"""

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

    draft, reflection = _parse_draft_and_reflection(raw)
    all_passed = reflection.get("all_passed", "NO") == "YES"
    failed_items = reflection.get("failed_items", [])

    print(f"  → Self-Reflection: {'통과 ✅' if all_passed else '미충족 항목 존재 ⚠️'}")
    if not all_passed:
        print(f"  → 미충족 항목: {failed_items}")

    state["draft"] = draft
    state["reflection_feedback"] = (
        f"미충족 항목: {failed_items}" if not all_passed else ""
    )
    state["draft_passed"] = all_passed
    state["retry_draft"] = retry_draft

    return state


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
    print(test_state["draft"])
    print(f"\n=== SC 통과 여부: {test_state['draft_passed']} ===")
    print(f"미충족 항목: {test_state['reflection_feedback']}")