# extractors/basic.py
from pathlib import Path
import fitz  # pymupdf
import docx
import pandas as pd

TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log"}
DOCX_EXTS = {".docx"}
EXCEL_EXTS = {".xlsx", ".xls"}

def extract_text_from_pdf(path: Path):
    try:
        doc = fitz.open(str(path))
        pages = []
        for p in doc:
            pages.append(p.get_text())
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(path: Path):
    try:
        d = docx.Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])
    except Exception:
        return ""

def extract_text_from_excel(path: Path):
    try:
        sheets = pd.read_excel(str(path), sheet_name=None)
        rows = []
        for name, df in sheets.items():
            rows.append(f"== SHEET: {name} ==")
            rows.append(df.astype(str).fillna("").to_csv(index=False))
        return "\n".join(rows)
    except Exception:
        return ""

def fallback_filename_meta(path: Path):
    try:
        return f"{path.stem} (type={path.suffix}, size={path.stat().st_size})"
    except Exception:
        return path.stem

def extract_text(path: Path):
    suf = path.suffix.lower()
    try:
        if suf in TEXT_EXTS:
            return path.read_text(encoding="utf-8", errors="ignore")
        elif suf == ".pdf":
            t = extract_text_from_pdf(path)
            return t if t.strip() else fallback_filename_meta(path)
        elif suf in DOCX_EXTS:
            t = extract_text_from_docx(path)
            return t if t.strip() else fallback_filename_meta(path)
        elif suf in EXCEL_EXTS:
            t = extract_text_from_excel(path)
            return t if t.strip() else fallback_filename_meta(path)
        else:
            return fallback_filename_meta(path)
    except Exception:
        return fallback_filename_meta(path)
