"""
Build a single Word document from the per-chapter thesis parts.

Reads the ordered Markdown parts under ``docs/thesis_parts/`` and produces one
merged Markdown file plus a ``.docx`` under ``docs/final_report/``. Figures are
embedded from ``docs/thesis_figures/`` and LaTeX math is converted to native
Word equations (OMML) so the file imports cleanly into Google Docs
(upload to Drive, then "Open with Google Docs").

Requires pandoc. If it is not on PATH, install the bundled binary with:
    pip install pypandoc-binary

Run from the project root:
    python scripts/build_thesis_docx.py
"""
import re
from pathlib import Path

import pypandoc

BASE = Path(__file__).resolve().parent.parent
PARTS_DIR = BASE / "docs" / "thesis_parts"
FIG_DIR = BASE / "docs" / "thesis_figures"
OUT_DIR = BASE / "docs" / "final_report"

# Order matters: front matter, the five chapters, then references.
PARTS = [
    "00_on_bolumler",
    "01_giris",
    "02_literatur_taramasi",
    "03_materyal_ve_yontem",
    "04_bulgular_ve_tartisma",
    "05_sonuc_ve_oneriler",
    "06_kaynaklar",
]

# Raw OpenXML hard page break, passed straight through to the .docx.
PAGE_BREAK = (
    '\n\n```{=openxml}\n'
    '<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'
    '```\n\n'
)


def _clean(md: str) -> str:
    """Strip internal HTML notes and normalise inline HTML for docx output."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)   # internal reviewer notes
    md = re.sub(r"</?div[^>]*>", "", md)              # unwrap right-aligned blocks
    md = re.sub(r"<br\s*/?>", "  \n", md)             # <br> -> Markdown line break
    return md.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = [_clean((PARTS_DIR / f"{name}.md").read_text(encoding="utf-8"))
              for name in PARTS]
    merged = PAGE_BREAK.join(chunks)

    merged_md = OUT_DIR / "thesis_merged.md"
    merged_md.write_text(merged, encoding="utf-8")

    out_docx = OUT_DIR / "Derin_Ogrenme_ile_Secici_Gurultu_Engelleme.docx"
    pypandoc.convert_file(
        str(merged_md), "docx", outputfile=str(out_docx),
        extra_args=[
            f"--resource-path={PARTS_DIR}",
            "--metadata=title:Derin Öğrenme ile Seçici Gürültü Engelleme",
            "--metadata=lang:tr",
        ],
    )
    print(f"Merged Markdown : {merged_md}")
    print(f"Word document   : {out_docx} ({out_docx.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
