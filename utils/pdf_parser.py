"""
utils/pdf_parser.py
Robust PDF -> plain text extraction for resumes.

Primary engine: pdfplumber (better layout / table handling)
Fallback engine: PyPDF2 (works when pdfplumber fails on odd encodings)

Both engines are wrapped in defensive try/except blocks so a corrupted or
password-protected PDF never crashes the Streamlit app -- it raises a clean
`PDFParsingError` with a human-readable message instead.
"""

import io
from typing import Union

import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed by any available engine."""
    pass


def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        if len(pdf.pages) == 0:
            raise PDFParsingError("The PDF has no pages.")
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def _extract_with_pypdf2(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise PDFParsingError(
                "This PDF is password-protected. Please upload an unprotected file."
            )
    text_chunks = []
    for page in reader.pages:
        text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks).strip()


def extract_text_from_pdf(file_input: Union[bytes, "io.BufferedIOBase"]) -> str:
    """
    Extract raw text from a PDF file.

    Args:
        file_input: raw bytes OR a file-like object (e.g. Streamlit's
                     UploadedFile) containing a PDF.

    Returns:
        Extracted plain text (str). Never returns None.

    Raises:
        PDFParsingError: if the file is empty, corrupted, encrypted without a
                          usable password, or contains no extractable text.
    """
    if file_input is None:
        raise PDFParsingError("No file was provided.")

    if hasattr(file_input, "read"):
        file_bytes = file_input.read()
    else:
        file_bytes = file_input

    if not file_bytes:
        raise PDFParsingError("The uploaded file is empty.")

    # Try pdfplumber first.
    text = ""
    plumber_error = None
    try:
        text = _extract_with_pdfplumber(file_bytes)
    except Exception as e:  # noqa: BLE001 - we deliberately fall back on ANY error
        plumber_error = e

    if text and text.strip():
        return text

    # Fall back to PyPDF2.
    try:
        text = _extract_with_pypdf2(file_bytes)
    except PdfReadError as e:
        raise PDFParsingError(f"Could not read PDF file: {e}")
    except PDFParsingError:
        raise
    except Exception as e:  # noqa: BLE001
        detail = f" (pdfplumber error: {plumber_error})" if plumber_error else ""
        raise PDFParsingError(f"Failed to parse PDF with both available engines: {e}{detail}")

    if not text or not text.strip():
        raise PDFParsingError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image-based resume -- please paste the text manually instead."
        )

    return text.strip()


def is_probably_resume(text: str) -> bool:
    """
    Lightweight heuristic sanity check that the extracted text looks like a
    resume rather than a random document. Not used to block processing --
    only for optional UI warnings.
    """
    if not text or len(text.strip()) < 50:
        return False
    keywords = ["experience", "education", "skills", "project", "工作",
                "university", "college", "certification", "summary", "objective"]
    lowered = text.lower()
    hits = sum(1 for kw in keywords if kw in lowered)
    return hits >= 1
