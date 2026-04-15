from dotenv import load_dotenv
load_dotenv()

MAX_RETRY_RAG = 2
MAX_RETRY_WEB = 2
MAX_RETRY_DRAFT = 2
MAX_LLM_CALLS = 20


def run_supervisor(state: dict) -> dict:
    print("\n[Supervisor] 상태 점검 중...")

    state.setdefault("retry_rag", 0)
    state.setdefault("retry_web", 0)
    state.setdefault("retry_draft", 0)
    state.setdefault("total_llm_calls", 0)
    state.setdefault("pdf_status", "")
    state.setdefault("draft", "")
    state.setdefault("draft_passed", False)
    state.setdefault("analysis_passed", False)
    state.setdefault("analysis_results", {})
    state.setdefault("rag_results", [])
    state.setdefault("web_results", [])

    _print_status(state)

    # 1. 비용 초과
    if state["total_llm_calls"] >= MAX_LLM_CALLS:
        print("  → END (LLM 호출 초과)")
        state["next"] = "END"
        return state

    # 2. PDF 완료
    if state["pdf_status"] in ("success", "fallback_md"):
        print(f"  → END ({state['pdf_status']})")
        state["next"] = "END"
        return state

    # 3. PDF 실패
    if state["pdf_status"] == "fail":
        print("  → formatting")
        state["next"] = "formatting"
        return state

    # 4. draft 통과
    if state["draft"] and state["draft_passed"]:
        print("  → formatting")
        state["next"] = "formatting"
        return state

    # 5. draft 있음 + 미통과
    if state["draft"] and not state["draft_passed"]:
        if state["retry_draft"] >= MAX_RETRY_DRAFT:
            print("  → formatting (강제)")
            state["next"] = "formatting"
            return state
        print(f"  → draft 재호출 (retry {state['retry_draft']}/{MAX_RETRY_DRAFT})")
        state["retry_draft"] += 1
        state["total_llm_calls"] += 1
        state["next"] = "draft"
        return state

    # ── 핵심 수정: analysis 실행 여부를 analysis_results로 판단 ──

    # 6. analysis 이미 실행됨 → draft로 진행 (통과/미통과 무관)
    if state["analysis_results"]:
        print("  → draft (analysis 결과 있음)")
        state["total_llm_calls"] += 1
        state["next"] = "draft"
        return state

    # 7. RAG + Web 둘 다 있음 → analysis 첫 실행
    if state["rag_results"] and state["web_results"]:
        print("  → analysis (첫 실행)")
        state["total_llm_calls"] += 1
        state["next"] = "analysis"
        return state

    # 8. RAG 있음, Web 없음
    if state["rag_results"] and not state["web_results"]:
        if state["retry_web"] >= MAX_RETRY_WEB:
            # 웹 없이 analysis 진행 (RAG만으로)
            print("  → analysis (Web 없이 RAG만으로 진행)")
            state["total_llm_calls"] += 1
            state["next"] = "analysis"
            return state
        print(f"  → web (retry {state['retry_web']}/{MAX_RETRY_WEB})")
        state["retry_web"] += 1
        state["next"] = "web"
        return state

    # 9. RAG 없음
    if state["retry_rag"] >= MAX_RETRY_RAG:
        print("  → web (RAG 초과)")
        state["next"] = "web"
        return state

    print(f"  → rag (retry {state['retry_rag']}/{MAX_RETRY_RAG})")
    state["retry_rag"] += 1
    state["next"] = "rag"
    return state


def _print_status(state: dict) -> None:
    print(f"  LLM 호출: {state['total_llm_calls']}/{MAX_LLM_CALLS}")
    print(f"  RAG: {len(state['rag_results'])}건 | Web: {len(state['web_results'])}건")
    print(f"  Analysis 실행: {'완료' if state['analysis_results'] else '미실행'}")
    print(f"  Draft: {'있음' if state['draft'] else '없음'} (통과: {state['draft_passed']})")
    print(f"  PDF: {state['pdf_status'] or '미실행'}")