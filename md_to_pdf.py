# scripts/md_to_pdf.py
# 사용법: python scripts/md_to_pdf.py outputs/파일명.md

import sys
import os
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors


# ── 한글 폰트 경로 (Mac 기준) ─────────────────────────────
# md_to_pdf.py 상단 FONT_PATHS 교체

FONT_PATHS_REGULAR = [
    os.path.expanduser("~/Library/Fonts/NanumGothic-Regular.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

FONT_PATHS_BOLD = [
    os.path.expanduser("~/Library/Fonts/NanumGothic-Bold.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]


def find_font() -> str:
    import glob
    for path in FONT_PATHS_REGULAR:
        matches = glob.glob(path)
        if matches:
            return matches[0]
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "한글 폰트를 찾을 수 없어요.\n"
    )


def register_fonts():
    font_path = find_font()
    print(f"  → 폰트 사용: {font_path}")
    pdfmetrics.registerFont(TTFont("Korean", font_path))
    pdfmetrics.registerFont(TTFont("KoreanBold", font_path))


def make_styles() -> dict:
    return {
        "h1":     ParagraphStyle("h1",     fontName="KoreanBold", fontSize=16, leading=20, spaceBefore=14, spaceAfter=8,  textColor=colors.HexColor("#1a1a2e")),
        "h2":     ParagraphStyle("h2",     fontName="KoreanBold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6,  textColor=colors.HexColor("#16213e")),
        "h3":     ParagraphStyle("h3",     fontName="KoreanBold", fontSize=11, leading=15, spaceBefore=8,  spaceAfter=4,  textColor=colors.HexColor("#0f3460")),
        "normal": ParagraphStyle("normal", fontName="Korean",     fontSize=9,  leading=15, spaceBefore=2,  spaceAfter=3),
        "bullet": ParagraphStyle("bullet", fontName="Korean",     fontSize=9,  leading=15, leftIndent=12,  spaceAfter=2),
        "quote":  ParagraphStyle("quote",  fontName="Korean",     fontSize=8,  leading=13, leftIndent=16,  spaceAfter=4,  textColor=colors.HexColor("#555555")),
    }


def safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_inline(text: str, font: str = "Korean") -> str:
    """**bold** → <b>bold</b> 변환"""
    text = safe(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    return text


def parse_table(lines: list, styles: dict) -> Table:
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
        ('FONTNAME',    (0,0), (-1,0),  'KoreanBold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Korean'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#16213e')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    return t


def md_to_story(md_text: str, styles: dict) -> list:
    story = []
    lines = md_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # 테이블 감지
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            t = parse_table(table_lines, styles)
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
            story.append(Paragraph(parse_inline(line[2:]), styles["quote"]))
            i += 1
            continue

        # 헤딩
        if line.startswith("# "):
            story.append(Paragraph(parse_inline(line[2:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(parse_inline(line[3:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(parse_inline(line[4:]), styles["h3"]))
            i += 1
            continue

        # 불릿
        if line.startswith("- ") or line.startswith("  - "):
            indent = len(line) - len(line.lstrip())
            text = line.strip()[2:]
            style = styles["bullet"]
            story.append(Paragraph("• " + parse_inline(text), style))
            i += 1
            continue

        # 일반 텍스트
        story.append(Paragraph(parse_inline(line), styles["normal"]))
        i += 1

    return story


def convert(md_path: str) -> str:
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"파일 없음: {md_path}")

    pdf_path = md_path.with_suffix(".pdf")

    print(f"[MD → PDF 변환]")
    print(f"  입력: {md_path}")
    print(f"  출력: {pdf_path}")

    register_fonts()
    styles = make_styles()

    md_text = md_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    story = md_to_story(md_text, styles)
    doc.build(story)

    print(f"  → 완료: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/md_to_pdf.py outputs/파일명.md")
        sys.exit(1)

    convert(sys.argv[1])