# agents/formatting_node.py
# 작성: 2반 이한결, 한채윤

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── 출력 경로 설정 ─────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_formatting(state: dict) -> dict:
    """
    Formatting Node
    - 검증 완료된 초안 → Markdown 파일로 저장
    - PDF 변환 시도 → 실패 시 Markdown Fallback
    - 이진 판단: success / fallback_md
    """
    print("\n[Formatting Node] 보고서 저장 시작...")

    draft = state.get("draft", "")
    if not draft:
        print("  ⚠️  초안 없음 → 저장 실패")
        state["pdf_status"] = "fail"
        return state

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"ai-mini_output_2반_이한결+한채윤_{timestamp}"

    # ── 1차: PDF 시도 ──────────────────────────────────────
    try:
        pdf_path = _save_as_pdf(draft, filename_base)
        print(f"  → PDF 저장 완료: {pdf_path}")
        state["pdf_status"] = "success"
        state["final_report_path"] = str(pdf_path)

    except Exception as e:
        print(f"  ⚠️  PDF 변환 실패: {e}")
        print(f"  → Markdown Fallback 실행...")

        # ── Fallback: Markdown 저장 ────────────────────────
        try:
            md_path = _save_as_markdown(draft, filename_base)
            print(f"  → Markdown 저장 완료: {md_path}")
            state["pdf_status"] = "fallback_md"
            state["final_report_path"] = str(md_path)

        except Exception as e2:
            print(f"  ⚠️  Markdown 저장도 실패: {e2}")
            state["pdf_status"] = "fail"
            state["final_report_path"] = ""

    return state


# ── PDF 저장 ──────────────────────────────────────────────
def _save_as_pdf(draft: str, filename_base: str) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ── 한글 폰트 등록 ──────────────────────────────────
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"폰트 없음: {font_path} → brew install font-nanum 또는 apt install fonts-nanum")
    
    pdfmetrics.registerFont(TTFont("NanumGothic", font_path))

    pdf_path = OUTPUT_DIR / f"{filename_base}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    
    # ── 한글 스타일 정의 ──────────────────────────────────
    h1 = ParagraphStyle("h1_kor", fontName="NanumGothic", fontSize=16, spaceAfter=10, spaceBefore=12)
    h2 = ParagraphStyle("h2_kor", fontName="NanumGothic", fontSize=13, spaceAfter=8, spaceBefore=10)
    h3 = ParagraphStyle("h3_kor", fontName="NanumGothic", fontSize=11, spaceAfter=6, spaceBefore=8)
    normal = ParagraphStyle("normal_kor", fontName="NanumGothic", fontSize=9, leading=15, spaceAfter=4)

    story = []

    for line in draft.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if line.startswith("# "):
            story.append(Paragraph(safe[2:], h1))
        elif line.startswith("## "):
            story.append(Paragraph(safe[3:], h2))
        elif line.startswith("### "):
            story.append(Paragraph(safe[4:], h3))
        else:
            story.append(Paragraph(safe, normal))

    doc.build(story)
    return pdf_path


# ── Markdown 저장 ─────────────────────────────────────────
def _save_as_markdown(draft: str, filename_base: str) -> Path:
    md_path = OUTPUT_DIR / f"{filename_base}.md"

    fallback_header = (
        "> ⚠️ PDF 변환 실패로 Markdown 형식으로 제공됩니다.\n\n"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(fallback_header + draft)

    return md_path


# ── 단독 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "draft": "# 테스트 보고서\n\n## SUMMARY\n\n테스트 내용입니다.\n\n## 1. 분석 배경\n\n테스트 배경입니다.",
        "pdf_status": "",
        "final_report_path": "",
    }

    result = run_formatting(test_state)
    print(f"\n상태: {result['pdf_status']}")
    print(f"저장 경로: {result['final_report_path']}")