import os
import re
import glob
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 폰트 경로 ─────────────────────────────────────────────
FONT_PATHS_REGULAR = [
    os.path.expanduser("~/Library/Fonts/NanumGothic-Regular.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
FONT_PATHS_BOLD = [
    os.path.expanduser("~/Library/Fonts/NanumGothic-Bold.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]

def _find_font(paths: list) -> str:
    for path in paths:
        matches = glob.glob(path)
        if matches:
            return matches[0]
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"폰트 없음: {paths}\n→ brew install --cask font-nanum-gothic")

def _register_fonts():
    regular = _find_font(FONT_PATHS_REGULAR)
    bold = _find_font(FONT_PATHS_BOLD)
    pdfmetrics.registerFont(TTFont("Korean", regular))
    pdfmetrics.registerFont(TTFont("KoreanBold", bold))

def _make_styles() -> dict:
    return {
        "h1":     ParagraphStyle("h1",     fontName="KoreanBold", fontSize=16, leading=20, spaceBefore=14, spaceAfter=8,  textColor=colors.HexColor("#1a1a2e")),
        "h2":     ParagraphStyle("h2",     fontName="KoreanBold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6,  textColor=colors.HexColor("#16213e")),
        "h3":     ParagraphStyle("h3",     fontName="KoreanBold", fontSize=11, leading=15, spaceBefore=8,  spaceAfter=4,  textColor=colors.HexColor("#0f3460")),
        "normal": ParagraphStyle("normal", fontName="Korean",     fontSize=9,  leading=15, spaceBefore=2,  spaceAfter=3),
        "bullet": ParagraphStyle("bullet", fontName="Korean",     fontSize=9,  leading=15, leftIndent=12,  spaceAfter=2),
        "quote":  ParagraphStyle("quote",  fontName="Korean",     fontSize=8,  leading=13, leftIndent=16,  spaceAfter=4,  textColor=colors.HexColor("#555555")),
    }

def _safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _parse_inline(text: str) -> str:
    text = _safe(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    return text

def _parse_table(lines: list) -> Table:
    rows = []
    for line in lines:
        if re.match(r'^\|[-| :]+\|$', line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return None
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTNAME',       (0,0), (-1,0),  'KoreanBold'),
        ('FONTNAME',       (0,1), (-1,-1), 'Korean'),
        ('FONTSIZE',       (0,0), (-1,-1), 8),
        ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor('#16213e')),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID',           (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',     (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 5),
        ('LEFTPADDING',    (0,0), (-1,-1), 6),
    ]))
    return t

def _md_to_story(md_text: str, styles: dict) -> list:
    story = []
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # 테이블
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            t = _parse_table(table_lines)
            if t:
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            continue

        # 구분선
        if re.match(r'^---+$', line.strip()):
            story.append(HRFlowable(width='100%', thickness=0.5,
                                     color=colors.HexColor('#cccccc'),
                                     spaceAfter=6, spaceBefore=6))
            i += 1
            continue

        # 빈 줄
        if not line.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        # 인용문
        if line.startswith("> "):
            story.append(Paragraph(_parse_inline(line[2:]), styles["quote"]))
            i += 1
            continue

        # 헤딩
        if line.startswith("# "):
            story.append(Paragraph(_parse_inline(line[2:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(_parse_inline(line[3:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(_parse_inline(line[4:]), styles["h3"]))
            i += 1
            continue

        # 불릿
        if line.startswith("- ") or line.startswith("  - "):
            text = line.strip()[2:]
            story.append(Paragraph("• " + _parse_inline(text), styles["bullet"]))
            i += 1
            continue

        # 일반
        story.append(Paragraph(_parse_inline(line), styles["normal"]))
        i += 1

    return story


# ── Formatting Node 실행 ──────────────────────────────────
def run_formatting(state: dict) -> dict:
    """
    Formatting Node
    - draft → PDF 변환 (한글 NanumGothic 폰트)
    - 실패 시 Markdown Fallback
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

    # ── PDF 시도 ──────────────────────────────────────────
    try:
        _register_fonts()
        styles = _make_styles()
        pdf_path = OUTPUT_DIR / f"{filename_base}.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=20*mm,  bottomMargin=20*mm,
        )

        story = _md_to_story(draft, styles)
        doc.build(story)

        print(f"  → PDF 저장 완료: {pdf_path}")
        state["pdf_status"] = "success"
        state["final_report_path"] = str(pdf_path)

    except Exception as e:
        print(f"  ⚠️  PDF 변환 실패: {e}")
        print(f"  → Markdown Fallback 실행...")

        try:
            md_path = OUTPUT_DIR / f"{filename_base}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("> ⚠️ PDF 변환 실패로 Markdown 형식으로 제공됩니다.\n\n")
                f.write(draft)
            print(f"  → Markdown 저장 완료: {md_path}")
            state["pdf_status"] = "fallback_md"
            state["final_report_path"] = str(md_path)

        except Exception as e2:
            print(f"  ⚠️  Markdown 저장도 실패: {e2}")
            state["pdf_status"] = "fail"
            state["final_report_path"] = ""

    return state


# ── 단독 실행 테스트 ──────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "draft": "# 테스트 보고서\n\n## SUMMARY\n\n테스트 내용입니다.\n\n## 1. 분석 배경\n\n- 항목 1\n- 항목 2",
        "pdf_status": "",
        "final_report_path": "",
    }
    result = run_formatting(test_state)
    print(f"\n상태: {result['pdf_status']}")
    print(f"경로: {result['final_report_path']}")