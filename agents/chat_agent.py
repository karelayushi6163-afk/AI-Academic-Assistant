"""
agents/chat_agent.py
The Career Coach Chat Agent -- ties together:
  1. A defined persona / system prompt (rubric: "Agent Prompting & Persona")
  2. RAG retrieval as a callable tool (rubric: "LangChain RAG Architecture")
  3. Native LLM tool-calling / function calling (rubric: "Tool Binding &
     Function Calling")
  4. Full conversational memory, since the Streamlit UI passes the entire
     chat_history back in on every turn (rubric: "User Interface" session
     memory)

This is the agent used by the "💬 AI Career Chat" tab in app.py.
"""

from utils.llm_client import LLMClient, LLMError
from utils.tools import CareerAgentTools

SYSTEM_PROMPT = """You are Aria, a professional AI Career Coach agent for the "Agentic AI Career \
Assessment & Skill Gap Analyzer" platform.

PERSONA & TONE:
- You are warm, encouraging, and direct -- like a senior mentor who wants the candidate to actually \
succeed, not just feel good.
- You give specific, actionable advice grounded in evidence, not generic motivational filler.
- You are honest about weaknesses without being discouraging.

SCOPE (stay within this -- politely decline anything unrelated):
- Career guidance, skill-gap analysis, learning roadmaps, interview preparation, and questions about \
the candidate's own resume/results for their target role: {target_role}.
- If asked something clearly outside this scope (e.g. general trivia, unrelated coding help, personal \
opinions on unrelated topics), politely redirect: "I'm focused on helping with your career assessment \
and job-readiness for {target_role} -- happy to help with anything in that space!"

HOW TO USE YOUR TOOLS:
- You have access to tools: search_career_knowledge (RAG retrieval over role guides + the candidate's \
resume + their analysis results), get_skill_gap_summary, get_roadmap_week, and \
get_interview_questions_by_topic.
- ALWAYS use search_career_knowledge when a question could be grounded in the candidate's resume, their \
results, or role-specific guidance -- do not answer from general knowledge alone when a tool could give \
you a grounded, specific answer.
- Use get_skill_gap_summary, get_roadmap_week, or get_interview_questions_by_topic when the user asks \
about those specific pieces of their own generated results.
- After calling a tool, synthesize the retrieved information into a natural, conversational answer -- \
never just dump raw tool output verbatim.
- If none of the tools return useful information, say so honestly rather than inventing details.

Keep responses concise (3-6 sentences unless the user asks for something longer, like a detailed plan).
"""


class ChatAgent:
    """The conversational Career Coach agent (RAG + tools + persona + memory)."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def respond(
        self,
        chat_history: list,
        target_role: str,
        gap_result: dict,
        roadmap: list,
        interview_questions: list,
        vector_store,
    ):
        """
        Args:
            chat_history: full conversation so far, list of
                {"role": "user"|"assistant", "content": str}. The LAST item
                should be the newest user message.
        Returns:
            (answer_text: str, tool_call_log: list[dict])
        """
        tools_handler = CareerAgentTools(
            gap_result=gap_result,
            roadmap=roadmap,
            interview_questions=interview_questions,
            target_role=target_role,
            vector_store=vector_store,
        )

        system = SYSTEM_PROMPT.format(target_role=target_role or "your target role")

        try:
            answer, tool_log = self.llm.chat_with_tools(
                system=system,
                chat_history=chat_history,
                tools_handler=tools_handler,
                max_tokens=1200,
                temperature=0.5,
            )
            return answer, tool_log
        except (LLMError, Exception) as e:  # noqa: BLE001
            return (
                f"Sorry, I couldn't process that right now ({e}). "
                f"Please check your API key in the sidebar and try again.",
                [],
            )
