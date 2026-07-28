"""
utils/report_generator.py
Builds the final downloadable report (Markdown and PDF) from whatever
analysis results are currently stored in Streamlit's session_state.
"""

from datetime import datetime
from io import BytesIO

from fpdf import FPDF


def _safe(value, default="N/A"):
    return value if value not in (None, "", []) else default


def build_markdown_report(profile: dict, gap_result: dict, roadmap: list,
                           interview_qs: list, target_role: str, experience_level: str) -> str:
    """Assemble a full Markdown report string from analysis results."""
    lines = []
    lines.append(f"# Career Assessment Report")
    lines.append(f"**Target Role:** {target_role}  ")
    lines.append(f"**Experience Level:** {experience_level}  ")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n---\n")

    # --- Profile ---
    lines.append("## 📄 Candidate Profile\n")
    if profile:
        lines.append(f"**Name:** {_safe(profile.get('name'))}")
        lines.append(f"**Experience Level (detected):** {_safe(profile.get('experience_level'))}")
        lines.append(f"**Years of Experience (approx.):** {_safe(profile.get('years_experience'))}\n")
        lines.append(f"**Summary:** {_safe(profile.get('summary'))}\n")

        skills = profile.get("skills") or []
        lines.append(f"**Extracted Skills ({len(skills)}):** " + ", ".join(skills) if skills else "**Extracted Skills:** None found")

        education = profile.get("education") or []
        if education:
            lines.append("\n**Education:**")
            for ed in education:
                lines.append(f"- {ed}")

        certs = profile.get("certifications") or []
        if certs:
            lines.append("\n**Certifications:**")
            for c in certs:
                lines.append(f"- {c}")
    else:
        lines.append("_No resume analyzed yet._")

    lines.append("\n---\n")

    # --- Gap Analysis ---
    lines.append("## 🎯 Skill Gap Analysis\n")
    if gap_result:
        lines.append(f"**Overall Match Score:** {gap_result.get('overall_score', 0):.1f}%\n")
        lines.append(gap_result.get("narrative", ""))

        lines.append("\n### Category Breakdown\n")
        lines.append("| Category | Match % |")
        lines.append("|---|---|")
        for cat, score in (gap_result.get("category_scores") or {}).items():
            lines.append(f"| {cat} | {score:.1f}% |")

        lines.append("\n### Matched Skills\n")
        matched = gap_result.get("matched_skills") or []
        lines.append(", ".join(matched) if matched else "_None_")

        lines.append("\n### Missing Skills (Critical)\n")
        critical = gap_result.get("missing_critical") or []
        lines.append(", ".join(critical) if critical else "_None_")

        lines.append("\n### Missing Skills (Nice-to-have)\n")
        nice = gap_result.get("missing_nice_to_have") or []
        lines.append(", ".join(nice) if nice else "_None_")
    else:
        lines.append("_No gap analysis run yet._")

    lines.append("\n---\n")

    # --- Roadmap ---
    lines.append("## 🗺️ Personalized Career Roadmap\n")
    if roadmap:
        for week in roadmap:
            lines.append(f"### Week {week.get('week')}: {week.get('focus')}")
            topics = week.get("topics") or []
            if topics:
                lines.append("**Topics:** " + ", ".join(topics))
            resources = week.get("resources") or []
            if resources:
                lines.append("\n**Resources:**")
                for r in resources:
                    lines.append(f"- {r}")
            if week.get("project"):
                lines.append(f"\n**Hands-on Project:** {week.get('project')}")
            if week.get("milestone"):
                lines.append(f"\n**Milestone:** {week.get('milestone')}")
            lines.append("")
    else:
        lines.append("_No roadmap generated yet._")

    lines.append("\n---\n")

    # --- Interview Prep ---
    lines.append("## 💡 Interview Preparation\n")
    if interview_qs:
        for i, q in enumerate(interview_qs, 1):
            lines.append(f"**Q{i}. ({q.get('category', 'General')} | {q.get('difficulty', 'Medium')}):** {q.get('question')}")
            lines.append(f"> **Answer Guideline:** {q.get('answer_guideline')}\n")
    else:
        lines.append("_No interview questions generated yet._")

    return "\n".join(lines)


def _to_latin1(text: str) -> str:
    """Encode to latin-1 safely (fpdf core fonts are latin-1 only)."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _break_long_tokens(text: str, max_token_len: int = 45) -> str:
    """
    Insert breakable spaces inside any unbroken "word" longer than
    max_token_len characters (e.g. long URLs, IDs, or run-on strings with no
    spaces). Without this, fpdf2's word-wrapper can find no place to break
    the line and raises 'Not enough horizontal space to render a single
    character' once the token is wider than the page.
    """
    out_words = []
    for word in text.split(" "):
        if len(word) > max_token_len:
            chunks = [word[i:i + max_token_len] for i in range(0, len(word), max_token_len)]
            out_words.append(" ".join(chunks))
        else:
            out_words.append(word)
    return " ".join(out_words)


def _safe_multi_cell(pdf: "FPDF", h: float, text: str) -> None:
    """
    Write a line with multi_cell, defensively handling any edge case that
    could otherwise raise (long unbroken tokens, zero-width edge cases,
    unsupported characters). Falls back to ever-safer renderings rather
    than ever letting the PDF export crash.
    """
    text = _to_latin1(_break_long_tokens(text))
    if not text.strip():
        pdf.ln(h / 2)
        return

    try:
        # wrapmode="CHAR" lets fpdf2 break mid-word if it must, instead of
        # raising when a "word" can't fit on one line.
        pdf.multi_cell(0, h, text, wrapmode="CHAR")
        return
    except Exception:
        pass

    # Fallback 1: hard-chunk the text ourselves into small fixed-width
    # pieces so every call is guaranteed to fit, regardless of content.
    try:
        chunk_size = 40
        for i in range(0, len(text), chunk_size):
            pdf.multi_cell(0, h, text[i:i + chunk_size])
        return
    except Exception:
        pass

    # Fallback 2: give up on this specific line rather than failing the
    # whole report -- note it and move on.
    try:
        pdf.multi_cell(0, h, "[unrenderable line omitted]")
    except Exception:
        pdf.ln(h)


def markdown_to_pdf_bytes(markdown_text: str, title: str = "Career Assessment Report") -> bytes:
    """
    Render a (lightly-formatted) markdown report into a downloadable PDF
    using fpdf2. This is a pragmatic line-based renderer -- it handles
    headings, bullet points, and plain paragraphs, which covers everything
    produced by build_markdown_report().

    Robust against long unbroken tokens (long URLs, IDs, etc.) that would
    otherwise crash fpdf2's word-wrapper -- see _safe_multi_cell().
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _safe_multi_cell(pdf, 10, title)
    pdf.ln(2)

    for raw_line in markdown_text.split("\n"):
        line = raw_line.rstrip()

        if not line.strip():
            pdf.ln(3)
            continue

        clean = (line.replace("**", "")
                     .replace("### ", "")
                     .replace("## ", "")
                     .replace("# ", "")
                     .replace("> ", "")
                     .replace("|", " "))
        clean = _to_latin1(clean)

        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            _safe_multi_cell(pdf, 9, clean)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(2)
            _safe_multi_cell(pdf, 8, clean)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            _safe_multi_cell(pdf, 7, clean)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            _safe_multi_cell(pdf, 6, f"  -  {clean[2:]}")
        elif line.startswith("---"):
            pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 10)
            _safe_multi_cell(pdf, 6, clean)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        output = output.encode("latin-1", "replace")
    return bytes(output)
