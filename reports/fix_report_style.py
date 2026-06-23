"""修改已有 docx 报告：统一所有字体颜色（含标题样式）为黑色，移除实验报告页眉。"""
from pathlib import Path
from docx import Document
from docx.shared import RGBColor


ROOT = Path(__file__).resolve().parents[1]
BLACK = RGBColor(0, 0, 0)

FILES = [
    (ROOT / "reports" / "experiment_report" / "深度学习实验报告.docx", True),
    (ROOT / "reports" / "course_design_report" / "深度学习课程设计报告.docx", False),
]


def fix_style_colors(doc):
    for style in doc.styles:
        if not hasattr(style, "font"):
            continue
        if style.font.color and style.font.color.rgb:
            style.font.color.rgb = BLACK


def fix_run_colors(doc):
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.color and run.font.color.rgb:
                run.font.color.rgb = BLACK

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.color and run.font.color.rgb:
                            run.font.color.rgb = BLACK

    for section in doc.sections:
        for paragraph in section.header.paragraphs:
            for run in paragraph.runs:
                if run.font.color and run.font.color.rgb:
                    run.font.color.rgb = BLACK
        for paragraph in section.footer.paragraphs:
            for run in paragraph.runs:
                if run.font.color and run.font.color.rgb:
                    run.font.color.rgb = BLACK


def remove_header(doc):
    for section in doc.sections:
        header = section.header
        header.is_linked_to_previous = False
        for paragraph in header.paragraphs:
            paragraph.clear()


def main():
    for filepath, remove_hdr in FILES:
        if not filepath.exists():
            print(f"skip (not found): {filepath}")
            continue
        print(f"processing: {filepath}")
        doc = Document(str(filepath))
        fix_style_colors(doc)
        fix_run_colors(doc)
        if remove_hdr:
            remove_header(doc)
            print("  -> header removed")
        doc.save(str(filepath))
        print("  -> saved")
    print("done")


if __name__ == "__main__":
    main()
