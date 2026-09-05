"""
utils/tools.py
Custom Python functions bound to the LLM via native tool-calling / function
calling (Anthropic's `tools` parameter / Groq's OpenAI-compatible `tools`
parameter). This is what "Tool Binding & Function Calling" in the rubric
refers to: the LLM itself decides, based on the user's message, WHEN to call
one of these functions and with WHAT arguments -- there is no manual
if/else keyword matching on our side.

Each tool is described by a JSON-schema spec (Anthropic format natively;
converted to OpenAI/Groq's format automatically) plus a Python
implementation that CareerAgentTools.execute() dispatches to.
"""

from typing import List


class CareerAgentTools:
    """
    Holds the current session's data (gap analysis, roadmap, interview
    questions, vector store) and exposes it as LLM-callable tools.
    """

    def __init__(self, gap_result: dict, roadmap: list, interview_questions: list,
                 target_role: str, vector_store=None):
        self.gap_result = gap_result or {}
        self.roadmap = roadmap or []
        self.interview_questions = interview_questions or []
        self.target_role = target_role
        self.vector_store = vector_store
        self.last_tool_calls: List[dict] = []  # for UI transparency

    # ------------------------------------------------------------------
    # Tool specs (Anthropic-native JSON schema format)
    # ------------------------------------------------------------------
    @property
    def specs(self) -> List[dict]:
        return [
            {
                "name": "search_career_knowledge",
                "description": (
                    "Search the RAG knowledge base (role guides, the candidate's resume, "
                    "and their analysis results) for information relevant to a query. "
                    "Use this whenever the user asks something that could be grounded in "
                    "the candidate's own resume/results or in role-specific career guidance, "
                    "instead of answering purely from general knowledge."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query -- what information to look up.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "How many relevant chunks to retrieve (default 4).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_skill_gap_summary",
                "description": (
                    "Get the candidate's precomputed skill-gap analysis for their target role "
                    "(overall match score, matched skills, missing critical skills, risk level). "
                    "Use this when the user asks about their score, readiness, or missing skills."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_roadmap_week",
                "description": (
                    "Get the detailed learning plan for a specific week of the candidate's "
                    "personalized roadmap. Use this when the user asks about a specific week "
                    "or what to study next."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "week_number": {
                            "type": "integer",
                            "description": "Which week of the roadmap to retrieve (1-indexed).",
                        }
                    },
                    "required": ["week_number"],
                },
            },
            {
                "name": "get_interview_questions_by_topic",
                "description": (
                    "Retrieve mock interview questions filtered by a topic/skill keyword or by "
                    "difficulty. Use this when the user asks to practice interview questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "A skill or category keyword to filter questions by (optional).",
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["Easy", "Medium", "Hard"],
                            "description": "Filter by difficulty (optional).",
                        },
                    },
                },
            },
        ]

    def specs_openai_format(self) -> List[dict]:
        """Convert Anthropic-style specs into OpenAI/Groq's `tools` schema."""
        converted = []
        for spec in self.specs:
            converted.append({
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            })
        return converted

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call to its implementation and return a string result."""
        self.last_tool_calls.append({"tool": tool_name, "input": tool_input})

        try:
            if tool_name == "search_career_knowledge":
                return self._search_career_knowledge(
                    tool_input.get("query", ""), tool_input.get("top_k", 4)
                )
            elif tool_name == "get_skill_gap_summary":
                return self._get_skill_gap_summary()
            elif tool_name == "get_roadmap_week":
                return self._get_roadmap_week(tool_input.get("week_number"))
            elif tool_name == "get_interview_questions_by_topic":
                return self._get_interview_questions_by_topic(
                    tool_input.get("topic"), tool_input.get("difficulty")
                )
            else:
                return f"Error: unknown tool '{tool_name}'."
        except Exception as e:  # noqa: BLE001
            return f"Error executing tool '{tool_name}': {e}"

    def _search_career_knowledge(self, query: str, top_k: int) -> str:
        if not self.vector_store or not query:
            return "No knowledge base is available to search right now."
        results = self.vector_store.similarity_search(query, k=top_k or 4)
        if not results:
            return "No relevant information found in the knowledge base for this query."
        formatted = []
        for r in results:
            formatted.append(f"[Source: {r['source']} | relevance={r['score']:.2f}]\n{r['text']}")
        return "\n\n---\n\n".join(formatted)

    def _get_skill_gap_summary(self) -> str:
        if not self.gap_result:
            return "No skill gap analysis has been run yet for this candidate."
        g = self.gap_result
        return (
            f"Target role: {self.target_role}\n"
            f"Overall match score: {g.get('overall_score', 0)}%\n"
            f"Risk level: {g.get('risk_level', 'N/A')}\n"
            f"Matched skills: {', '.join(g.get('matched_skills', [])) or 'None'}\n"
            f"Missing critical skills: {', '.join(g.get('missing_critical_combined', [])) or 'None'}\n"
            f"Top priorities: {', '.join(g.get('top_priorities', [])) or 'None'}\n"
            f"Narrative: {g.get('narrative', '')}"
        )

    def _get_roadmap_week(self, week_number) -> str:
        if not self.roadmap:
            return "No roadmap has been generated yet for this candidate."
        try:
            week_number = int(week_number)
        except (TypeError, ValueError):
            return "Please specify a valid week number."
        for week in self.roadmap:
            if week.get("week") == week_number:
                return (
                    f"Week {week.get('week')}: {week.get('focus')}\n"
                    f"Topics: {', '.join(week.get('topics', []))}\n"
                    f"Resources: {', '.join(week.get('resources', []))}\n"
                    f"Project: {week.get('project', '')}\n"
                    f"Milestone: {week.get('milestone', '')}"
                )
        return f"Week {week_number} is not part of the generated roadmap (it has {len(self.roadmap)} weeks)."

    def _get_interview_questions_by_topic(self, topic, difficulty) -> str:
        if not self.interview_questions:
            return "No interview questions have been generated yet for this candidate."
        filtered = self.interview_questions
        if topic:
            topic_lower = topic.lower()
            filtered = [
                q for q in filtered
                if topic_lower in (q.get("targets_skill", "") or "").lower()
                or topic_lower in (q.get("question", "") or "").lower()
            ]
        if difficulty:
            filtered = [q for q in filtered if q.get("difficulty", "").lower() == difficulty.lower()]

        if not filtered:
            return f"No interview questions matched topic='{topic}', difficulty='{difficulty}'."

        formatted = []
        for q in filtered[:5]:
            formatted.append(
                f"Q: {q.get('question')}\n"
                f"Category: {q.get('category')} | Difficulty: {q.get('difficulty')}\n"
                f"Answer guideline: {q.get('answer_guideline')}"
            )
        return "\n\n".join(formatted)
