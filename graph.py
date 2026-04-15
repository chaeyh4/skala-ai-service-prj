# graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.rag_agent import get_rag_agent
from agents.web_search_agent import run_web_search
from agents.analysis_agent import run_analysis
from agents.draft_agent import run_draft
from agents.formatting_node import run_formatting
from agents.supervisor import run_supervisor

rag_agent = get_rag_agent()

# ── State 정의 ────────────────────────────────────────────
class State(TypedDict):
    query: str
    rag_results: list
    web_results: list
    sc_scores: dict
    analysis_results: dict
    analysis_passed: bool
    draft: str
    draft_passed: bool
    reflection_feedback: str
    retry_rag: int
    retry_web: int
    retry_analysis: int
    retry_draft: int
    total_llm_calls: int
    next: str
    pdf_status: str
    final_report_path: str
    limitation_summary: str
    source_count: int

# ── 노드 함수 래퍼 ────────────────────────────────────────
from agents.rag_agent import get_rag_agent
from agents.web_search_agent import run_web_search
from agents.analysis_agent import run_analysis
from agents.draft_agent import run_draft
from agents.formatting_node import run_formatting
from agents.supervisor import run_supervisor

rag_agent = get_rag_agent()

def node_rag(state: State) -> State:
    output = rag_agent.run(state["query"])
    state["rag_results"] = output["rag_results"]
    return state

def node_web(state: State) -> State:
    return run_web_search(state)

def node_analysis(state: State) -> State:
    return run_analysis(state)

def node_draft(state: State) -> State:
    return run_draft(state)

def node_formatting(state: State) -> State:
    return run_formatting(state)

def node_supervisor(state: State) -> State:
    return run_supervisor(state)

# ── Conditional Edge 함수 ─────────────────────────────────
def route(state: State) -> str:
    return state.get("next", "END")

# ── Graph 구성 ────────────────────────────────────────────
def build_graph():
    g = StateGraph(State)

    # 노드 등록
    g.add_node("supervisor", node_supervisor)
    g.add_node("rag",        node_rag)
    g.add_node("web",        node_web)
    g.add_node("analysis",   node_analysis)
    g.add_node("draft",      node_draft)
    g.add_node("formatting", node_formatting)

    # 시작점
    g.set_entry_point("supervisor")

    # 모든 worker → supervisor 보고
    g.add_edge("rag",        "supervisor")
    g.add_edge("web",        "supervisor")
    g.add_edge("analysis",   "supervisor")
    g.add_edge("draft",      "supervisor")
    g.add_edge("formatting", "supervisor")

    # supervisor → conditional edge로 다음 노드 결정
    g.add_conditional_edges(
        "supervisor",
        route,
        {
            "rag":        "rag",
            "web":        "web",
            "analysis":   "analysis",
            "draft":      "draft",
            "formatting": "formatting",
            "END":        END,
        }
    )

    return g.compile()

graph = build_graph()

if __name__ == "__main__":
    from langchain_core.runnables.graph import MermaidDrawMethod
    
    # 그래프 시각화
    png = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )
    with open("outputs/graph.png", "wb") as f:
        f.write(png)
    print("graph.png 저장 완료")