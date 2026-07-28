"""
agents/roadmap_agent.py
Career Roadmap Agent.

Responsibility:
    Given the skill gap analysis for a target role, generate a structured,
    week-by-week learning roadmap: topics to study, recommended resources
    (courses/docs/books), a hands-on project, and a milestone for each week.
"""

from typing import List

from utils.llm_client import LLMClient, LLMError, extract_json

SYSTEM_PROMPT = """You are an expert technical curriculum designer and career
mentor. You design realistic, actionable, week-by-week learning roadmaps that
a busy working professional could actually follow. You respond ONLY with
valid JSON -- no markdown fences, no commentary outside the JSON."""

USER_PROMPT_TEMPLATE = """Design a {weeks}-week personalized learning roadmap for a candidate targeting the role: "{target_role}".

Candidate experience level: {experience_level}
Candidate's current matched skills: {matched_skills}
Candidate's CRITICAL missing skills to prioritize: {missing_critical}
Candidate's nice-to-have missing skills: {missing_nice_to_have}

Return ONLY a JSON array (length exactly {weeks}) with this schema per item:

[
  {{
    "week": 1,
    "focus": "<short theme for the week, e.g. 'Python & Data Fundamentals'>",
    "topics": ["<topic 1>", "<topic 2>", "<topic 3>"],
    "resources": ["<specific resource: course name, official docs, book, or platform (e.g. 'DeepLearning.AI - Neural Networks and Deep Learning course')>", "..."],
    "project": "<one concrete hands-on mini-project for this week that reinforces the topics>",
    "milestone": "<a measurable outcome/checkpoint to know the week was successful>",
    "estimated_hours": <integer estimated study hours for the week>
  }}
]

Rules:
- Order weeks so foundational/critical gaps are addressed first, then build toward more advanced/nice-to-have skills.
- Keep resource names realistic and well-known (official docs, Coursera, freeCodeCamp, Udemy, YouTube channels, O'Reilly books, Kaggle, LeetCode, etc.) -- do not invent fake course titles.
- Make projects progressively build on each other where possible, culminating in something portfolio-worthy by the final week.
- Tailor pacing and depth to the stated experience level.
"""


class RoadmapAgent:
    """Agent responsible for generating a personalized learning roadmap."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        gap_result: dict,
        target_role: str,
        experience_level: str,
        weeks: int = 8,
    ) -> List[dict]:
        matched = gap_result.get("matched_skills", [])
        missing_critical = gap_result.get("missing_critical_combined") or gap_result.get("missing_critical", [])
        missing_nice = gap_result.get("missing_nice_to_have", [])

        prompt = USER_PROMPT_TEMPLATE.format(
            weeks=weeks,
            target_role=target_role,
            experience_level=experience_level,
            matched_skills=", ".join(matched) or "None yet",
            missing_critical=", ".join(missing_critical) or "None -- focus on deepening existing skills",
            missing_nice_to_have=", ".join(missing_nice) or "None",
        )

        try:
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, max_tokens=3000, temperature=0.5)
            data = extract_json(raw)
            if not isinstance(data, list):
                raise LLMError("Roadmap response was not a JSON array.")
            return self._normalize(data, weeks)
        except (LLMError, Exception) as e:  # noqa: BLE001
            return self._fallback_roadmap(missing_critical, missing_nice, weeks, str(e))

    @staticmethod
    def _normalize(data: list, weeks: int) -> list:
        normalized = []
        for i, item in enumerate(data[:weeks], 1):
            normalized.append({
                "week": item.get("week", i),
                "focus": item.get("focus", f"Week {i}"),
                "topics": item.get("topics", []) or [],
                "resources": item.get("resources", []) or [],
                "project": item.get("project", ""),
                "milestone": item.get("milestone", ""),
                "estimated_hours": item.get("estimated_hours", 8),
            })
        return normalized

    @staticmethod
    def _fallback_roadmap(missing_critical: list, missing_nice: list, weeks: int, error: str) -> list:
        """Deterministic fallback roadmap if the LLM call fails entirely."""
        all_gaps = (missing_critical or []) + (missing_nice or []) or ["Core role fundamentals"]
        roadmap = []
        chunk_size = max(1, len(all_gaps) // weeks + 1)
        for i in range(weeks):
            chunk = all_gaps[i * chunk_size: (i + 1) * chunk_size] or [all_gaps[i % len(all_gaps)]]
            roadmap.append({
                "week": i + 1,
                "focus": f"Deep dive: {', '.join(chunk)}",
                "topics": chunk,
                "resources": [f"Official documentation and a structured course for: {t}" for t in chunk],
                "project": f"Build a small project applying {', '.join(chunk)}.",
                "milestone": f"Comfortably explain and apply {', '.join(chunk)} in an interview setting.",
                "estimated_hours": 8,
                "_warning": f"AI roadmap generation failed ({error}); showing a basic fallback plan." if i == 0 else None,
            })
        return roadmap
