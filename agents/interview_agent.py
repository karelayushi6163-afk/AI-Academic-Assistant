"""
agents/interview_agent.py
Interview Prep Agent.

Responsibility:
    Generate tailored mock technical interview questions -- weighted toward
    the candidate's weakest/missing skill areas for the target role -- along
    with concise answer guidelines (not full scripted answers, so the user
    still has to think) to practice with.
"""

from typing import List

from utils.llm_client import LLMClient, LLMError, extract_json

SYSTEM_PROMPT = """You are a senior technical interviewer who has hired for
top technology companies. You design realistic, role-appropriate interview
questions that probe both a candidate's existing strengths and their weaker
areas. You respond ONLY with valid JSON -- no markdown fences, no commentary
outside the JSON."""

USER_PROMPT_TEMPLATE = """Create {num_questions} mock interview questions for a candidate targeting the role: "{target_role}" at experience level "{experience_level}".

Candidate's matched/strong skills: {matched_skills}
Candidate's CRITICAL missing skills (weight questions toward probing awareness/learning-ability here): {missing_critical}

Return ONLY a JSON array (length exactly {num_questions}) with this schema per item:

[
  {{
    "question": "<the interview question>",
    "category": "<one of: Technical, Behavioral, System Design, Problem Solving>",
    "difficulty": "<one of: Easy, Medium, Hard>",
    "targets_skill": "<the specific skill this question is designed to probe>",
    "answer_guideline": "<3-4 sentence guideline on what a strong answer should cover -- NOT a full scripted answer, but the key points/structure to hit>"
  }}
]

Rules:
- Roughly 60% of questions should target the candidate's missing/weak skills (to help them prepare), and 40% should reinforce their existing strengths.
- Include a healthy mix of categories, not just pure technical trivia.
- Calibrate difficulty to the stated experience level.
- Make questions specific and realistic, not generic ("Tell me about yourself" is too generic -- prefer role-specific behavioral questions instead).
"""


class InterviewAgent:
    """Agent responsible for generating tailored mock interview questions."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        gap_result: dict,
        target_role: str,
        experience_level: str,
        num_questions: int = 8,
    ) -> List[dict]:
        matched = gap_result.get("matched_skills", [])
        missing_critical = gap_result.get("missing_critical_combined") or gap_result.get("missing_critical", [])

        prompt = USER_PROMPT_TEMPLATE.format(
            num_questions=num_questions,
            target_role=target_role,
            experience_level=experience_level,
            matched_skills=", ".join(matched) or "General fundamentals",
            missing_critical=", ".join(missing_critical) or "None -- focus on advanced/edge-case scenarios",
        )

        try:
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, max_tokens=2500, temperature=0.6)
            data = extract_json(raw)
            if not isinstance(data, list):
                raise LLMError("Interview question response was not a JSON array.")
            return self._normalize(data, num_questions)
        except (LLMError, Exception) as e:  # noqa: BLE001
            return self._fallback_questions(matched, missing_critical, num_questions, str(e))

    @staticmethod
    def _normalize(data: list, num_questions: int) -> list:
        normalized = []
        for item in data[:num_questions]:
            normalized.append({
                "question": item.get("question", ""),
                "category": item.get("category", "Technical"),
                "difficulty": item.get("difficulty", "Medium"),
                "targets_skill": item.get("targets_skill", ""),
                "answer_guideline": item.get("answer_guideline", ""),
            })
        return normalized

    @staticmethod
    def _fallback_questions(matched: list, missing_critical: list, num_questions: int, error: str) -> list:
        """Deterministic fallback question bank if the LLM call fails entirely."""
        pool_skills = (missing_critical or [])[:5] + (matched or [])[:5]
        if not pool_skills:
            pool_skills = ["your target role's core fundamentals"]

        questions = []
        for i in range(num_questions):
            skill = pool_skills[i % len(pool_skills)]
            is_weak = skill in (missing_critical or [])
            questions.append({
                "question": (
                    f"Walk me through how you would approach learning and applying '{skill}' "
                    f"in a real project, given you're less experienced there."
                    if is_weak else
                    f"Describe a time you used '{skill}' to solve a challenging problem. "
                    f"What tradeoffs did you consider?"
                ),
                "category": "Technical",
                "difficulty": "Medium",
                "targets_skill": skill,
                "answer_guideline": (
                    f"Discuss a structured approach: understand fundamentals, build a small project, "
                    f"and identify how '{skill}' fits into the broader system. Mention resources or "
                    f"communities you'd use to get up to speed quickly."
                    if is_weak else
                    f"Describe context, your specific role, the technical decision process, and the "
                    f"measurable outcome. Emphasize depth of understanding of '{skill}'."
                ),
                "_warning": f"AI interview question generation failed ({error}); showing fallback questions." if i == 0 else None,
            })
        return questions
