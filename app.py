# app.py
from graph import graph

def main():
    initial_state = {
        "query": "HBM4 Hybrid Bonding 기술에 대한 SK하이닉스, Samsung, TSMC의 R&D 전략과 TRL을 분석해줘",
        "rag_results": [],
        "web_results": [],
        "sc_scores": {},
        "analysis_results": {},
        "analysis_passed": False,
        "draft": "",
        "draft_passed": False,
        "reflection_feedback": "",
        "retry_rag": 0,
        "retry_web": 0,
        "retry_analysis": 0,
        "retry_draft": 0,
        "total_llm_calls": 0,
        "next": "",
        "pdf_status": "",
        "final_report_path": "",
        "limitation_summary": "",
        "source_count": 0,
    }

    print("=== HBM4 Hybrid Bonding 기술 전략 분석 시작 ===\n")
    result = graph.invoke(initial_state)

    print(f"\n=== 완료 ===")
    print(f"PDF 상태: {result['pdf_status']}")
    print(f"저장 경로: {result['final_report_path']}")

if __name__ == "__main__":
    main()