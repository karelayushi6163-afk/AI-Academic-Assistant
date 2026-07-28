"""
agents/gap_analyzer.py
Market Gap Analyzer Agent.

Responsibility:
    Compare the candidate's extracted skills against the deterministic
    ROLE_SKILL_MATRIX (config.py) for the chosen target role, compute a
    reproducible numeric match score per category and overall, and then ask
    the LLM to produce a qualitative narrative + prioritized recommendations
    on top of that deterministic scaffold.

Design note:
    The numeric scoring NEVER depends on the LLM -- it uses fuzzy string
    matching (difflib) against the static skill matrix. This guarantees a
    consistent, explainable score even if the LLM call fails, and the LLM is
    only used to add human-readable insight on top.
"""

import difflib
from typing import List

from config import ROLE_SKILL_MATRIX, get_role_skill_matrix
from utils.llm_client import LLMClient, LLMError, extract_json

SYSTEM_PROMPT = """You are a senior technical career coach and industry hiring
analyst. You give honest, specific, encouraging but realistic feedback about
skill gaps for a target job role. You respond ONLY with valid JSON -- no
markdown fences, no commentary outside the JSON."""

USER_PROMPT_TEMPLATE = """A candidate is targeting the role: "{target_role}" at experience level "{experience_level}".

Candidate's current skills: {current_skills}

Skills the candidate is MISSING that are considered CRITICAL for this role:
{missing_critical}

Skills the candidate is MISSING that are NICE-TO-HAVE for this role:
{missing_nice_to_have}

Skills the candidate ALREADY HAS that match this role:
{matched_skills}

Overall computed match score: {overall_score:.1f}%

Return ONLY a JSON object with EXACTLY this schema:

{{
  "narrative": "<3-5 sentence honest assessment of readiness for this role, referencing the score and biggest gaps>",
  "top_priorities": ["<most important skill/action to fix first>", "<second>", "<third>"],
  "strengths_summary": "<1-2 sentences on the candidate's strongest existing assets for this role>",
  "risk_level": "<one of: Low, Medium, High>"
}}
"""


def _fuzzy_match(skill_a: str, skill_b: str) -> bool:
    """Case-insensitive fuzzy match to tolerate spelling/format variance."""
    a, b = skill_a.lower().strip(), skill_b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.85


class GapAnalyzerAgent:
    """Agent responsible for computing and explaining skill gaps vs. a target role."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def analyze(self, current_skills: List[str], target_role: str, experience_level: str) -> dict:
        role_matrix = get_role_skill_matrix(target_role)
        if not role_matrix:
            raise ValueError(
                f"Unknown target role '{target_role}'. Please choose a role from the sidebar list."
            )

        current_skills_clean = [s.strip() for s in current_skills if s and s.strip()]

        matched_skills = []
        missing_critical = []
        missing_important = []
        missing_nice_to_have = []
        category_scores = {}

        for category, skill_weights in role_matrix.items():
            cat_total_weight = sum(skill_weights.values())
            cat_earned_weight = 0

            for skill, weight in skill_weights.items():
                is_matched = any(_fuzzy_match(skill, cs) for cs in current_skills_clean)
                if is_matched:
                    matched_skills.append(skill)
                    cat_earned_weight += weight
                else:
                    if weight == 3:
                        missing_critical.append(skill)
                    elif weight == 2:
                        missing_important.append(skill)
                    else:
                        missing_nice_to_have.append(skill)

            category_scores[category] = (cat_earned_weight / cat_total_weight * 100) if cat_total_weight else 0.0

        overall_earned = sum(
            weight for cat in role_matrix.values() for skill, weight in cat.items()
            if any(_fuzzy_match(skill, cs) for cs in current_skills_clean)
        )
        overall_total = sum(weight for cat in role_matrix.values() for weight in cat.values())
        overall_score = (overall_earned / overall_total * 100) if overall_total else 0.0

        result = {
            "target_role": target_role,
            "overall_score": round(overall_score, 1),
            "category_scores": {k: round(v, 1) for k, v in category_scores.items()},
            "matched_skills": sorted(set(matched_skills)),
            "missing_critical": sorted(set(missing_critical)),
            "missing_important": sorted(set(missing_important)),
            "missing_nice_to_have": sorted(set(missing_nice_to_have)),
            "extra_skills": [
                s for s in current_skills_clean
                if not any(_fuzzy_match(s, req_skill) for cat in role_matrix.values() for req_skill in cat)
            ],
        }

        # Combine critical + important into a single "missing_critical" bucket
        # for simpler UI display, while keeping the detailed breakdown too.
        result["missing_critical_combined"] = sorted(set(missing_critical + missing_important))

        # --- LLM narrative layer (best-effort; never blocks the numeric result) ---
        try:
            prompt = USER_PROMPT_TEMPLATE.format(
                target_role=target_role,
                experience_level=experience_level,
                current_skills=", ".join(current_skills_clean) or "None listed",
                missing_critical=", ".join(missing_critical) or "None",
                missing_nice_to_have=", ".join(missing_important + missing_nice_to_have) or "None",
                matched_skills=", ".join(matched_skills) or "None",
                overall_score=overall_score,
            )
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, max_tokens=800, temperature=0.4)
            llm_data = extract_json(raw)
            result["narrative"] = llm_data.get("narrative", "")
            result["top_priorities"] = llm_data.get("top_priorities", [])
            result["strengths_summary"] = llm_data.get("strengths_summary", "")
            result["risk_level"] = llm_data.get("risk_level", self._fallback_risk(overall_score))
        except (LLMError, Exception) as e:  # noqa: BLE001
            result["narrative"] = self._fallback_narrative(overall_score, missing_critical)
            result["top_priorities"] = (missing_critical + missing_important)[:3]
            result["strengths_summary"] = f"You already match {len(matched_skills)} required skills for this role."
            result["risk_level"] = self._fallback_risk(overall_score)
            result["_warning"] = f"AI narrative generation failed ({e}); showing a rule-based summary instead."

        return result

    @staticmethod
    def _fallback_risk(score: float) -> str:
        if score >= 70:
            return "Low"
        elif score >= 40:
            return "Medium"
        return "High"

    @staticmethod
    def _fallback_narrative(score: float, missing_critical: list) -> str:
        gap_text = ", ".join(missing_critical[:5]) if missing_critical else "no major gaps"
        return (
            f"Your current skill match for this role is approximately {score:.1f}%. "
            f"The most significant gaps are: {gap_text}. Focus your learning plan on "
            f"these areas first to maximize interview readiness."
        )


def list_available_roles() -> List[str]:
    return list(ROLE_SKILL_MATRIX.keys())
