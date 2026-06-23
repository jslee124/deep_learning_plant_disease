"""将两份 Markdown 报告生成为 A4 Word 正文版。"""

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    (
        ROOT / "reports" / "experiment_report" / "深度学习实验报告.md",
        ROOT / "reports" / "experiment_report" / "深度学习实验报告.docx",
        "深度学习实验报告",
        False,
    ),
    (
        ROOT / "reports" / "course_design_report" / "深度学习课程设计报告.md",
        ROOT / "reports" / "course_design_report" / "深度学习课程设计报告.docx",
        "深度学习课程设计报告",
        True,
    ),
]

CONTENT_WIDTH_DXA = 9026
LIGHT_FILL = "EEF4F3"


def set_run_font(run, east_asia="宋体", latin="Times New Roman", size=10.5, bold=None, color=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "100")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])
    set_run_font(run, size=9)


def configure_document(doc, running_title, add_header=True):
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    heading_tokens = {
        "Heading 1": (16, 12, 8),
        "Heading 2": (13, 10, 6),
        "Heading 3": (11.5, 8, 4),
    }
    for style_name, (size, before, after) in heading_tokens.items():
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if add_header:
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header_run = header.add_run(running_title)
        set_run_font(header_run, east_asia="黑体", latin="Arial", size=9)

    add_page_number(section.footer.paragraphs[0])


def add_inline_runs(paragraph, text, default_size=10.5):
    tokens = re.split(r"(\*\*.+?\*\*|`.+?`)", text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size=default_size, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, east_asia="等线", latin="Consolas", size=default_size - 0.5)
        else:
            run = paragraph.add_run(token)
            set_run_font(run, size=default_size)


def add_title(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(24)
    paragraph.paragraph_format.space_after = Pt(18)
    run = paragraph.add_run(text)
    set_run_font(run, east_asia="黑体", latin="Arial", size=22, bold=True)


def add_note(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.5)
    paragraph.paragraph_format.right_indent = Cm(0.5)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(10)
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), LIGHT_FILL)
    p_pr.append(shd)
    run = paragraph.add_run(text)
    set_run_font(run, size=9.5)


def parse_table(lines):
    rows = []
    for line in lines:
        values = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(values)
    return [rows[0], *rows[2:]]


def table_widths(column_count):
    ratios = {
        2: [0.38, 0.62],
        3: [0.42, 0.29, 0.29],
        5: [0.31, 0.17, 0.17, 0.18, 0.17],
        6: [0.25, 0.14, 0.14, 0.16, 0.16, 0.15],
        7: [0.25, 0.12, 0.12, 0.11, 0.14, 0.13, 0.13],
    }.get(column_count, [1 / column_count] * column_count)
    widths = [round(CONTENT_WIDTH_DXA * ratio) for ratio in ratios]
    widths[-1] += CONTENT_WIDTH_DXA - sum(widths)
    return widths


def add_table(doc, rows):
    column_count = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    set_table_geometry(table, table_widths(column_count))
    font_size = 8.2 if column_count >= 6 else 9.2

    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.15
            add_inline_runs(paragraph, value, default_size=font_size)
            if row_index == 0:
                set_cell_shading(cell, LIGHT_FILL)
                for run in paragraph.runs:
                    run.bold = True

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_image(doc, source_md, alt_text, relative_path):
    image_path = (source_md.parent / relative_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"报告图片不存在: {image_path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.05))
    doc.core_properties.comments = f"包含图片：{alt_text}"


def build_report(source_md, output_docx, running_title, add_header=True):
    text = source_md.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()
    configure_document(doc, running_title, add_header)

    index = 0
    title_used = False
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        content = " ".join(line.strip() for line in paragraph_buffer)
        paragraph = doc.add_paragraph()
        if re.match(r"^图\s*\d+", content):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(8)
            add_inline_runs(paragraph, content, default_size=9)
        else:
            paragraph.paragraph_format.first_line_indent = Cm(0.74)
            add_inline_runs(paragraph, content)
        paragraph_buffer = []

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped == r"\newpage":
            flush_paragraph()
            doc.add_page_break()
            index += 1
            continue

        image_match = re.match(r"!\[(.+?)\]\((.+?)\)", stripped)
        if image_match:
            flush_paragraph()
            add_image(doc, source_md, image_match.group(1), image_match.group(2))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            flush_paragraph()
            table_lines = [stripped, lines[index + 1].strip()]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            add_table(doc, parse_table(table_lines))
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            if level == 1 and not title_used:
                add_title(doc, heading_text)
                title_used = True
            else:
                paragraph = doc.add_heading(level=min(level, 3))
                add_inline_runs(paragraph, heading_text, default_size={1: 16, 2: 13, 3: 11.5}[level])
            index += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph()
            add_note(doc, stripped[2:])
            index += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.left_indent = Cm(0.74)
            paragraph.paragraph_format.first_line_indent = Cm(-0.37)
            paragraph.paragraph_format.space_after = Pt(4)
            add_inline_runs(paragraph, stripped[2:])
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_docx)
    print(f"已生成: {output_docx}")


def main():
    for source_md, output_docx, running_title, add_header in REPORTS:
        build_report(source_md, output_docx, running_title, add_header)


if __name__ == "__main__":
    main()
