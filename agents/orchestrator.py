"""
agents/orchestrator.py
LangGraph-based orchestration of the 4-agent pipeline:

    Resume Parser -> Gap Analyzer -> Roadmap Agent -> Interview Agent

Each tab in the Streamlit UI can also call its agent directly for a fast,
isolated action (e.g. "just re-run interview questions"). This orchestrator
is used for the "Run Full Pipeline" one-click flow, and demonstrates a real
LangGraph StateGraph wiring the agents together with shared state.

If the `langgraph` package is unavailable for any reason, `run_pipeline()`
transparently falls back to plain sequential execution of the same agents,
so the app never breaks because of the orchestration layer.
"""

from typing import Any, Dict, List, TypedDict

from agents.gap_analyzer import GapAnalyzerAgent
from agents.interview_agent import InterviewAgent
from agents.resume_agent import ResumeAgent
from agents.roadmap_agent import RoadmapAgent
from utils.llm_client import LLMClient


class PipelineState(TypedDict, total=False):
    resume_text: str
    target_role: str
    experience_level: str
    roadmap_weeks: int
    num_questions: int
    profile: Dict[str, Any]
    gap_result: Dict[str, Any]
    roadmap: List[Dict[str, Any]]
    interview_questions: List[Dict[str, Any]]
    errors: List[str]


def _make_nodes(llm_client: LLMClient):
    resume_agent = ResumeAgent(llm_client)
    gap_agent = GapAnalyzerAgent(llm_client)
    roadmap_agent = RoadmapAgent(llm_client)
    interview_agent = InterviewAgent(llm_client)

    def parse_resume_node(state: PipelineState) -> PipelineState:
        try:
            state["profile"] = resume_agent.extract(state["resume_text"])
        except Exception as e:  # noqa: BLE001
            state.setdefault("errors", []).append(f"Resume parsing failed: {e}")
            state["profile"] = {"skills": [], "experience_level": state.get("experience_level", "Entry-Level")}
        return state

    def gap_analysis_node(state: PipelineState) -> PipelineState:
        try:
            skills = state["profile"].get("skills", [])
            state["gap_result"] = gap_agent.analyze(
                skills, state["target_role"], state["experience_level"]
            )
        except Exception as e:  # noqa: BLE001
            state.setdefault("errors", []).append(f"Gap analysis failed: {e}")
            state["gap_result"] = {}
        return state

    def roadmap_node(state: PipelineState) -> PipelineState:
        try:
            state["roadmap"] = roadmap_agent.generate(
                state["gap_result"], state["target_role"], state["experience_level"],
                weeks=state.get("roadmap_weeks", 8),
            )
        except Exception as e:  # noqa: BLE001
            state.setdefault("errors", []).append(f"Roadmap generation failed: {e}")
            state["roadmap"] = []
        return state

    def interview_node(state: PipelineState) -> PipelineState:
        try:
            state["interview_questions"] = interview_agent.generate(
                state["gap_result"], state["target_role"], state["experience_level"],
                num_questions=state.get("num_questions", 8),
            )
        except Exception as e:  # noqa: BLE001
            state.setdefault("errors", []).append(f"Interview prep generation failed: {e}")
            state["interview_questions"] = []
        return state

    return parse_resume_node, gap_analysis_node, roadmap_node, interview_node


def build_graph(llm_client: LLMClient):
    """Build and compile the LangGraph StateGraph for the full pipeline."""
    from langgraph.graph import END, StateGraph

    parse_resume_node, gap_analysis_node, roadmap_node, interview_node = _make_nodes(llm_client)

    graph = StateGraph(PipelineState)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("interview_prep", interview_node)

    graph.set_entry_point("parse_resume")
    graph.add_edge("parse_resume", "gap_analysis")
    graph.add_edge("gap_analysis", "roadmap")
    graph.add_edge("roadmap", "interview_prep")
    graph.add_edge("interview_prep", END)

    return graph.compile()


def run_pipeline(
    llm_client: LLMClient,
    resume_text: str,
    target_role: str,
    experience_level: str,
    roadmap_weeks: int = 8,
    num_questions: int = 8,
) -> PipelineState:
    """
    Run the full 4-agent pipeline end-to-end. Tries LangGraph first; falls
    back to plain sequential execution if LangGraph is not installed or
    fails to compile for any reason.
    """
    initial_state: PipelineState = {
        "resume_text": resume_text,
        "target_role": target_role,
        "experience_level": experience_level,
        "roadmap_weeks": roadmap_weeks,
        "num_questions": num_questions,
        "errors": [],
    }

    try:
        compiled_graph = build_graph(llm_client)
        final_state = compiled_graph.invoke(initial_state)
        return final_state
    except Exception:  # noqa: BLE001 - fall back to manual sequential execution
        parse_resume_node, gap_analysis_node, roadmap_node, interview_node = _make_nodes(llm_client)
        state = initial_state
        state = parse_resume_node(state)
        state = gap_analysis_node(state)
        state = roadmap_node(state)
        state = interview_node(state)
        return state
