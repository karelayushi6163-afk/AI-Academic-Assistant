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


def _wrap_line(pdf: "FPDF", text: str, max_width: float) -> list:
    """
    Manually wrap `text` into a list of lines that each fit within
    `max_width`, using only pdf.get_string_width() -- fpdf2's most basic,
    stable measurement API. This deliberately avoids fpdf2's built-in
    multi_cell() word-wrap engine, which is the component that raises
    'Not enough horizontal space to render a single character' when it
    encounters a long unbroken token (e.g. a URL) it can't break cleanly.

    Any single word wider than max_width is itself broken down
    character-by-character, so this function can NEVER raise on width
    grounds -- worst case it produces a very short line.
    """
    if not text:
        return [""]

    lines = []
    current = ""

    for word in text.split(" "):
        if word == "":
            continue

        if pdf.get_string_width(word) > max_width:
            # This single word alone is too wide -- flush what we have,
            # then break the word itself into safe character chunks.
            if current:
                lines.append(current)
                current = ""
            chunk = ""
            for ch in word:
                trial = chunk + ch
                if pdf.get_string_width(trial) > max_width and chunk:
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk = trial
            current = chunk
            continue

        trial = f"{current} {word}".strip() if current else word
        if pdf.get_string_width(trial) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial

    if current:
        lines.append(current)

    return lines or [""]


def _write_line(pdf: "FPDF", h: float, text: str) -> None:
    """
    Write one logical line of text to the PDF, manually wrapping it to fit
    the page width first. Uses only pdf.cell() to draw each wrapped
    sub-line -- cell() does not attempt any internal wrapping/measurement
    that could raise, it simply draws the given text.
    """
    text = _to_latin1(text)
    if not text.strip():
        pdf.ln(h / 2)
        return

    max_width = pdf.w - pdf.l_margin - pdf.r_margin
    for sub_line in _wrap_line(pdf, text, max_width):
        # Manual page-break check + plain cell() + explicit ln() -- this
        # combination works identically across old and new fpdf2 releases,
        # unlike the newer new_x/new_y keyword args on cell().
        if pdf.get_y() + h > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, h, sub_line)
        pdf.ln(h)


def markdown_to_pdf_bytes(markdown_text: str, title: str = "Career Assessment Report") -> bytes:
    """
    Render a (lightly-formatted) markdown report into a downloadable PDF
    using fpdf2. This is a pragmatic line-based renderer -- it handles
    headings, bullet points, and plain paragraphs, which covers everything
    produced by build_markdown_report().

    Deliberately bypasses fpdf2's multi_cell() word-wrap engine (see
    _wrap_line / _write_line) to avoid version-specific edge cases in that
    engine that can raise 'Not enough horizontal space to render a single
    character' on long unbroken tokens like URLs.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _write_line(pdf, 10, title)
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

        try:
            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                _write_line(pdf, 9, clean)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.ln(2)
                _write_line(pdf, 8, clean)
            elif line.startswith("### "):
                pdf.set_font("Helvetica", "B", 11)
                _write_line(pdf, 7, clean)
            elif line.startswith("- "):
                pdf.set_font("Helvetica", "", 10)
                _write_line(pdf, 6, f"  -  {clean[2:]}")
            elif line.startswith("---"):
                pdf.ln(1)
            else:
                pdf.set_font("Helvetica", "", 10)
                _write_line(pdf, 6, clean)
        except Exception:
            # Absolute last resort: never let one malformed line take down
            # the whole export.
            pdf.set_x(pdf.l_margin)
            try:
                pdf.cell(0, 6, "[line omitted due to rendering error]", new_x="LMARGIN", new_y="NEXT")
            except Exception:
                pdf.ln(6)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        output = output.encode("latin-1", "replace")
    return bytes(output)
