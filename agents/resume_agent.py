"""
agents/resume_agent.py
Resume Parser & Skill Extractor Agent.

Responsibility:
    Take raw resume text (from PDF upload or pasted text) and return a
    structured profile: name, extracted technical skills, education,
    certifications, detected experience level, and a short summary.

The agent is LLM-driven (Claude/Groq via LLMClient) but always falls back to
a deterministic keyword-based extractor if the LLM call fails, so the app
never hard-crashes on a missing/invalid API key.
"""

import re
from typing import Optional

from utils.llm_client import LLMClient, LLMError, extract_json

SYSTEM_PROMPT = """You are an expert technical resume parser and career analyst.
You extract structured, factual information from resumes. You NEVER invent
skills, companies, or dates that are not present or strongly implied by the
text. You respond ONLY with valid JSON -- no markdown fences, no commentary."""

USER_PROMPT_TEMPLATE = """Analyze the following resume text and extract structured information.

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

Return ONLY a JSON object with EXACTLY this schema (no extra keys, no markdown fences):

{{
  "name": "<candidate's full name if present, else 'Candidate'>",
  "summary": "<2-3 sentence objective professional summary of this candidate>",
  "skills": ["<technical skill 1>", "<technical skill 2>", "..."],
  "soft_skills": ["<soft skill 1>", "..."],
  "education": ["<degree, institution, year if available>", "..."],
  "certifications": ["<certification name>", "..."],
  "experience_level": "<one of: Student/Fresher, Entry-Level, Mid-Level, Senior, Lead/Principal>",
  "years_experience": <approximate total number of years of professional experience as a number>,
  "past_roles": ["<job title 1>", "<job title 2>", "..."],
  "domains": ["<industry/domain the candidate has worked in, e.g. Fintech, Healthcare>"]
}}

Rules:
- "skills" should ONLY include concrete technical/tools/frameworks/languages skills (e.g. Python, React, AWS, Docker, SQL). Be comprehensive but avoid duplicates and avoid vague terms like "technology".
- If a field cannot be determined, use an empty list [] or an empty string "", or 0 for years_experience.
- Do not fabricate information not present in the resume text.
"""


class ResumeAgent:
    """Agent responsible for turning raw resume text into a structured profile."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract(self, resume_text: str) -> dict:
        if not resume_text or not resume_text.strip():
            raise ValueError("Resume text is empty. Please upload a PDF or paste text.")

        # Guard against extremely long resumes blowing the context window.
        trimmed_text = resume_text.strip()[:15000]

        try:
            prompt = USER_PROMPT_TEMPLATE.format(resume_text=trimmed_text)
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, max_tokens=2000, temperature=0.2)
            data = extract_json(raw)
            return self._normalize(data)
        except (LLMError, Exception) as e:  # noqa: BLE001
            # Deterministic fallback so the app remains usable without a
            # working LLM call (e.g. bad key, network issue, rate limit).
            fallback = self._fallback_extract(trimmed_text)
            fallback["_warning"] = (
                f"AI extraction failed ({e}); showing a basic keyword-based "
                f"extraction instead. Results may be less accurate."
            )
            return fallback

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(data: dict) -> dict:
        """Ensure all expected keys exist with sane types/defaults."""
        normalized = {
            "name": data.get("name") or "Candidate",
            "summary": data.get("summary") or "",
            "skills": [s.strip() for s in (data.get("skills") or []) if isinstance(s, str) and s.strip()],
            "soft_skills": [s.strip() for s in (data.get("soft_skills") or []) if isinstance(s, str) and s.strip()],
            "education": [e for e in (data.get("education") or []) if isinstance(e, str)],
            "certifications": [c for c in (data.get("certifications") or []) if isinstance(c, str)],
            "experience_level": data.get("experience_level") or "Entry-Level",
            "years_experience": _safe_number(data.get("years_experience")),
            "past_roles": [r for r in (data.get("past_roles") or []) if isinstance(r, str)],
            "domains": [d for d in (data.get("domains") or []) if isinstance(d, str)],
        }
        # De-duplicate skills case-insensitively while preserving order.
        seen = set()
        deduped = []
        for s in normalized["skills"]:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        normalized["skills"] = deduped
        return normalized

    # ------------------------------------------------------------------
    # Deterministic fallback extractor (no LLM required)
    # ------------------------------------------------------------------
    COMMON_SKILLS = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "R", "SQL",
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI", "Spring Boot",
        "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "Pandas", "NumPy", "Machine Learning",
        "Deep Learning", "NLP", "Computer Vision", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
        "Git", "Jenkins", "CI/CD", "Terraform", "Ansible", "MongoDB", "PostgreSQL", "MySQL",
        "Redis", "GraphQL", "REST APIs", "Microservices", "System Design", "HTML", "CSS",
        "Tailwind CSS", "Excel", "Tableau", "Power BI", "Statistics", "Linux", "Bash",
        "Network Security", "Cryptography", "Penetration Testing", "Wireshark", "Nmap",
        "Agile/Scrum", "Product Strategy", "A/B Testing",
    ]

    def _fallback_extract(self, text: str) -> dict:
        found_skills = []
        lowered = text.lower()
        for skill in self.COMMON_SKILLS:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
            if re.search(pattern, lowered):
                found_skills.append(skill)

        years_match = re.search(r"(\d+)\+?\s*years?\s+(of\s+)?experience", lowered)
        years = int(years_match.group(1)) if years_match else 0

        if years >= 8:
            level = "Lead/Principal"
        elif years >= 5:
            level = "Senior"
        elif years >= 2:
            level = "Mid-Level"
        elif years >= 1:
            level = "Entry-Level"
        else:
            level = "Student/Fresher"

        first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
        name_match = re.match(r"^([A-Z][a-zA-Z.\-]+(?:\s+[A-Z][a-zA-Z.\-]+){1,2})", first_line)
        name = name_match.group(1) if name_match else "Candidate"

        return {
            "name": name,
            "summary": "Automatically extracted via keyword matching (AI parsing unavailable).",
            "skills": found_skills,
            "soft_skills": [],
            "education": [],
            "certifications": [],
            "experience_level": level,
            "years_experience": years,
            "past_roles": [],
            "domains": [],
        }


def _safe_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
