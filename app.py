"""
app.py
Agentic AI Career Assessment & Skill Gap Analyzer -- Streamlit Dashboard.

Run locally with:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.gap_analyzer import GapAnalyzerAgent
from agents.interview_agent import InterviewAgent
from agents.resume_agent import ResumeAgent
from agents.roadmap_agent import RoadmapAgent
from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_ANTHROPIC_API_KEY,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_GROQ_MODEL,
    EXPERIENCE_LEVELS,
    GROQ_MODEL_LLAMA_70B,
    INTERVIEW_DEFAULT_QUESTIONS,
    ANTHROPIC_MODEL_HAIKU,
    ANTHROPIC_MODEL_SONNET,
    ROADMAP_DEFAULT_WEEKS,
    TARGET_ROLES,
)
from utils.llm_client import LLMClient, LLMError
from utils.pdf_parser import PDFParsingError, extract_text_from_pdf
from utils.report_generator import build_markdown_report, markdown_to_pdf_bytes

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def _init_state():
    defaults = {
        "resume_text": "",
        "profile": None,
        "gap_result": None,
        "roadmap": None,
        "interview_questions": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# --------------------------------------------------------------------------
# Sidebar: API keys, target role, experience level
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("#### 🔑 API Keys")
    provider_choice = st.radio(
        "Primary LLM Provider",
        options=["Anthropic (Claude)", "Groq (Llama 3.3)"],
        index=0,
        help="The other provider will be used automatically as a fallback if this one fails and its key is also provided.",
    )

    anthropic_key_input = st.text_input(
        "Anthropic API Key",
        value=DEFAULT_ANTHROPIC_API_KEY,
        type="password",
        placeholder="sk-ant-...",
    )
    groq_key_input = st.text_input(
        "Groq API Key",
        value=DEFAULT_GROQ_API_KEY,
        type="password",
        placeholder="gsk_...",
    )

    anthropic_model_choice = st.selectbox(
        "Claude Model",
        options=[ANTHROPIC_MODEL_SONNET, ANTHROPIC_MODEL_HAIKU],
        index=0,
        help="Sonnet = higher quality. Haiku = faster/cheaper.",
    )

    st.divider()
    st.markdown("#### 🎯 Target Role & Experience")
    target_role = st.selectbox("Target Job Role", options=TARGET_ROLES, index=0)
    experience_level = st.selectbox("Your Experience Level", options=EXPERIENCE_LEVELS, index=1)

    st.divider()
    st.markdown("#### 🗺️ Roadmap Settings")
    roadmap_weeks = st.slider("Roadmap Duration (weeks)", min_value=4, max_value=16, value=ROADMAP_DEFAULT_WEEKS)
    num_interview_qs = st.slider("Number of Interview Questions", min_value=4, max_value=15, value=INTERVIEW_DEFAULT_QUESTIONS)

    st.divider()
    if not anthropic_key_input and not groq_key_input:
        st.warning("⚠️ Enter at least one API key to run the AI agents.")
    else:
        st.success("✅ API key configured.")


def get_llm_client() -> LLMClient:
    provider_order = (
        ["anthropic", "groq"] if provider_choice.startswith("Anthropic") else ["groq", "anthropic"]
    )
    return LLMClient(
        anthropic_api_key=anthropic_key_input or None,
        groq_api_key=groq_key_input or None,
        anthropic_model=anthropic_model_choice,
        groq_model=GROQ_MODEL_LLAMA_70B,
        provider_order=provider_order,
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title(APP_TITLE)
st.caption(
    "A multi-agent AI system that parses your resume, benchmarks it against "
    "real market skill requirements, and builds you a personalized roadmap "
    "and interview prep plan."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Resume Upload & Profile Breakdown",
    "🎯 Skill Gap Analysis",
    "🗺️ Personalized Career Roadmap",
    "💡 Interview Prep & Practice",
])

# ==========================================================================
# TAB 1 — Resume Upload & Profile Breakdown
# ==========================================================================
with tab1:
    st.subheader("📄 Upload Your Resume")
    col_upload, col_paste = st.columns(2)

    with col_upload:
        uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"])
        if uploaded_file is not None:
            try:
                extracted_text = extract_text_from_pdf(uploaded_file)
                st.session_state["resume_text"] = extracted_text
                st.success(f"✅ Extracted {len(extracted_text)} characters from PDF.")
            except Exception as e:
                import traceback
                st.exception(e)
                st.code(traceback.format_exc())

    with col_paste:
        pasted_text = st.text_area(
            "...or paste your resume text here",
            value=st.session_state["resume_text"] if not uploaded_file else "",
            height=200,
            placeholder="Paste your resume content here (used if no PDF is uploaded above)...",
        )
        if pasted_text and not uploaded_file:
            st.session_state["resume_text"] = pasted_text

    analyze_col1, analyze_col2 = st.columns([1, 3])
    with analyze_col1:
        run_resume_analysis = st.button("🔍 Analyze Resume", type="primary", width='content')

    if run_resume_analysis:
        if not st.session_state["resume_text"].strip():
            st.error("Please upload a PDF or paste resume text first.")
        else:
            with st.spinner("🤖 Resume Parser Agent is extracting your skills, education, and experience..."):
                try:
                    llm_client = get_llm_client()
                    agent = ResumeAgent(llm_client)
                    profile = agent.extract(st.session_state["resume_text"])
                    st.session_state["profile"] = profile
                    if profile.get("_warning"):
                        st.warning(profile["_warning"])
                    else:
                        st.success("✅ Resume analyzed successfully!")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Resume analysis failed: {e}")

    profile = st.session_state.get("profile")
    if profile:
        st.divider()
        st.subheader(f"👤 Profile Breakdown: {profile.get('name', 'Candidate')}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Detected Experience Level", profile.get("experience_level", "N/A"))
        c2.metric("Approx. Years of Experience", profile.get("years_experience", 0))
        c3.metric("Skills Extracted", len(profile.get("skills", [])))

        if profile.get("summary"):
            st.info(profile["summary"])

        st.markdown("##### 🛠️ Technical Skills")
        skills = profile.get("skills", [])
        if skills:
            skill_html = " ".join(
                f"<span style='background-color:#1f77b4;color:white;padding:4px 10px;"
                f"border-radius:14px;margin:3px;display:inline-block;font-size:0.85em'>{s}</span>"
                for s in skills
            )
            st.markdown(skill_html, unsafe_allow_html=True)
        else:
            st.write("No technical skills detected.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 🎓 Education")
            education = profile.get("education", [])
            if education:
                for ed in education:
                    st.write(f"- {ed}")
            else:
                st.write("_None detected._")

            st.markdown("##### 📜 Certifications")
            certs = profile.get("certifications", [])
            if certs:
                for c in certs:
                    st.write(f"- {c}")
            else:
                st.write("_None detected._")

        with col_b:
            st.markdown("##### 💼 Past Roles")
            roles = profile.get("past_roles", [])
            if roles:
                for r in roles:
                    st.write(f"- {r}")
            else:
                st.write("_None detected._")

            st.markdown("##### 🌐 Domains / Industries")
            domains = profile.get("domains", [])
            if domains:
                for d in domains:
                    st.write(f"- {d}")
            else:
                st.write("_None detected._")

            if profile.get("soft_skills"):
                st.markdown("##### 🤝 Soft Skills")
                st.write(", ".join(profile["soft_skills"]))
    else:
        st.info("👆 Upload or paste your resume, then click **Analyze Resume** to see your profile breakdown.")


# ==========================================================================
# TAB 2 — Skill Gap Analysis
# ==========================================================================
with tab2:
    st.subheader(f"🎯 Skill Gap Analysis vs. {target_role}")

    if not st.session_state.get("profile"):
        st.info("Please analyze your resume in Tab 1 first.")
    else:
        run_gap_analysis = st.button("📊 Run Gap Analysis", type="primary")

        if run_gap_analysis:
            with st.spinner("🤖 Market Gap Analyzer Agent is comparing your skills to market requirements..."):
                try:
                    llm_client = get_llm_client()
                    agent = GapAnalyzerAgent(llm_client)
                    skills = st.session_state["profile"].get("skills", [])
                    gap_result = agent.analyze(skills, target_role, experience_level)
                    st.session_state["gap_result"] = gap_result
                    if gap_result.get("_warning"):
                        st.warning(gap_result["_warning"])
                    else:
                        st.success("✅ Gap analysis complete!")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Gap analysis failed: {e}")

        gap_result = st.session_state.get("gap_result")
        if gap_result:
            st.divider()

            score = gap_result.get("overall_score", 0)
            risk = gap_result.get("risk_level", "Medium")
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "🟡")

            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Match Score", f"{score:.1f}%")
            m2.metric("Readiness Risk", f"{risk_color} {risk}")
            m3.metric("Missing Critical Skills", len(gap_result.get("missing_critical_combined", [])))

            st.progress(min(int(score), 100) / 100)

            if gap_result.get("narrative"):
                st.markdown("##### 📝 Assessment")
                st.write(gap_result["narrative"])

            if gap_result.get("strengths_summary"):
                st.markdown("##### 💪 Strengths")
                st.write(gap_result["strengths_summary"])

            if gap_result.get("top_priorities"):
                st.markdown("##### 🚨 Top Priorities")
                for i, p in enumerate(gap_result["top_priorities"], 1):
                    st.write(f"{i}. {p}")

            st.divider()
            st.markdown("##### 📈 Category Match Breakdown")
            category_scores = gap_result.get("category_scores", {})
            if category_scores:
                fig = go.Figure(data=go.Scatterpolar(
                    r=list(category_scores.values()),
                    theta=list(category_scores.keys()),
                    fill='toself',
                    name='Your Match %',
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 🧩 Detailed Skill Matrix")
            all_rows = []
            for s in gap_result.get("matched_skills", []):
                all_rows.append({"Skill": s, "Status": "✅ Have It"})
            for s in gap_result.get("missing_critical", []):
                all_rows.append({"Skill": s, "Status": "❌ Missing (Critical)"})
            for s in gap_result.get("missing_important", []):
                all_rows.append({"Skill": s, "Status": "⚠️ Missing (Important)"})
            for s in gap_result.get("missing_nice_to_have", []):
                all_rows.append({"Skill": s, "Status": "➕ Missing (Nice-to-have)"})

            if all_rows:
                df = pd.DataFrame(all_rows).sort_values("Status")
                st.dataframe(df, width='stretch', hide_index=True)

            if gap_result.get("extra_skills"):
                with st.expander("✨ Skills you have that go beyond this role's core requirements"):
                    st.write(", ".join(gap_result["extra_skills"]))
        else:
            st.info("Click **Run Gap Analysis** above to see your skill match matrix for this role.")


# ==========================================================================
# TAB 3 — Personalized Career Roadmap
# ==========================================================================
with tab3:
    st.subheader(f"🗺️ Your {roadmap_weeks}-Week Roadmap to {target_role}")

    if not st.session_state.get("gap_result"):
        st.info("Please run the Skill Gap Analysis in Tab 2 first.")
    else:
        run_roadmap = st.button("🗺️ Generate Roadmap", type="primary")

        if run_roadmap:
            with st.spinner("🤖 Career Roadmap Agent is building your personalized learning path..."):
                try:
                    llm_client = get_llm_client()
                    agent = RoadmapAgent(llm_client)
                    roadmap = agent.generate(
                        st.session_state["gap_result"], target_role, experience_level, weeks=roadmap_weeks
                    )
                    st.session_state["roadmap"] = roadmap
                    if roadmap and roadmap[0].get("_warning"):
                        st.warning(roadmap[0]["_warning"])
                    else:
                        st.success("✅ Roadmap generated!")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Roadmap generation failed: {e}")

        roadmap = st.session_state.get("roadmap")
        if roadmap:
            st.divider()
            total_hours = sum(w.get("estimated_hours", 0) for w in roadmap)
            st.caption(f"Estimated total study time: **{total_hours} hours** across {len(roadmap)} weeks.")

            for week in roadmap:
                header = f"Week {week.get('week')}: {week.get('focus', '')}"
                with st.expander(header, expanded=(week.get('week') == 1)):
                    if week.get("topics"):
                        st.markdown("**📚 Topics to Cover:**")
                        for t in week["topics"]:
                            st.checkbox(t, key=f"topic_{week.get('week')}_{t}")

                    if week.get("resources"):
                        st.markdown("**🔗 Recommended Resources:**")
                        for r in week["resources"]:
                            st.write(f"- {r}")

                    if week.get("project"):
                        st.markdown("**🛠️ Hands-on Project:**")
                        st.write(week["project"])

                    if week.get("milestone"):
                        st.markdown("**🏁 Milestone:**")
                        st.success(week["milestone"])

                    st.caption(f"⏱️ Estimated effort: {week.get('estimated_hours', 'N/A')} hours")
        else:
            st.info("Click **Generate Roadmap** above to build your personalized learning path.")


# ==========================================================================
# TAB 4 — Interview Prep & Practice
# ==========================================================================
with tab4:
    st.subheader(f"💡 Mock Interview Prep for {target_role}")

    if not st.session_state.get("gap_result"):
        st.info("Please run the Skill Gap Analysis in Tab 2 first.")
    else:
        run_interview_prep = st.button("💡 Generate Interview Questions", type="primary")

        if run_interview_prep:
            with st.spinner("🤖 Interview Prep Agent is crafting tailored questions..."):
                try:
                    llm_client = get_llm_client()
                    agent = InterviewAgent(llm_client)
                    questions = agent.generate(
                        st.session_state["gap_result"], target_role, experience_level,
                        num_questions=num_interview_qs,
                    )
                    st.session_state["interview_questions"] = questions
                    if questions and questions[0].get("_warning"):
                        st.warning(questions[0]["_warning"])
                    else:
                        st.success("✅ Interview questions ready!")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Interview prep generation failed: {e}")

        questions = st.session_state.get("interview_questions")
        if questions:
            st.divider()

            categories = sorted(set(q.get("category", "Technical") for q in questions))
            selected_categories = st.multiselect("Filter by category", options=categories, default=categories)

            filtered = [q for q in questions if q.get("category", "Technical") in selected_categories]

            for i, q in enumerate(filtered, 1):
                diff_emoji = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(q.get("difficulty", "Medium"), "🟡")
                with st.expander(f"Q{i}. {q.get('question', '')}"):
                    st.caption(
                        f"**Category:** {q.get('category', 'Technical')} &nbsp;|&nbsp; "
                        f"**Difficulty:** {diff_emoji} {q.get('difficulty', 'Medium')} &nbsp;|&nbsp; "
                        f"**Targets Skill:** {q.get('targets_skill', 'N/A')}"
                    )
                    st.text_area("Your practice answer (not saved):", key=f"practice_{i}", height=100)
                    if st.button("Show Answer Guideline", key=f"show_ans_{i}"):
                        st.info(q.get("answer_guideline", ""))
        else:
            st.info("Click **Generate Interview Questions** above to start practicing.")


# ==========================================================================
# EXPORT SECTION (always visible at bottom)
# ==========================================================================
st.divider()
st.subheader("📤 Export Full Report")

has_any_data = any([
    st.session_state.get("profile"),
    st.session_state.get("gap_result"),
    st.session_state.get("roadmap"),
    st.session_state.get("interview_questions"),
])

if not has_any_data:
    st.caption("Run at least one analysis above to enable report export.")
else:
    markdown_report = build_markdown_report(
        profile=st.session_state.get("profile") or {},
        gap_result=st.session_state.get("gap_result") or {},
        roadmap=st.session_state.get("roadmap") or [],
        interview_qs=st.session_state.get("interview_questions") or [],
        target_role=target_role,
        experience_level=experience_level,
    )

    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
    with exp_col1:
        st.download_button(
            "⬇️ Download as Markdown",
            data=markdown_report,
            file_name="career_assessment_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp_col2:
        try:
            pdf_bytes = markdown_to_pdf_bytes(markdown_report, title="Career Assessment Report")
            st.download_button(
                "⬇️ Download as PDF",
                data=pdf_bytes,
                file_name="career_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"PDF generation error: {e}")

    with st.expander("👀 Preview Report"):
        st.markdown(markdown_report)

st.divider()
st.caption(
    "Built with a 4-agent architecture (Resume Parser → Gap Analyzer → Roadmap Agent → Interview Agent) "
    "orchestrated via LangGraph, powered by Anthropic Claude / Groq Llama. "
    "No resume data is stored server-side — everything lives only in this browser session."
)
