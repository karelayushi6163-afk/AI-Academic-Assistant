"""
utils/knowledge_base.py
The document corpus used by the RAG pipeline (utils/vector_store.py).

Each entry is a "document" in the RAG sense: a chunk-able block of text with
a source label. In this app the corpus is made of two kinds of documents:

  1. Static role-guide documents (below) -- one detailed write-up per target
     role covering responsibilities, expected tech stack, typical projects,
     and what interviewers actually probe for. These are what let the chat
     agent answer grounded questions like "why is Docker important for an
     AI/ML Engineer?" instead of guessing.

  2. The candidate's own resume text + their live analysis results (gap
     analysis narrative, roadmap, interview questions) -- added at runtime
     from app.py once those agents have run, so the chat agent can also
     answer questions grounded in the user's OWN data ("why did I get a
     high score in the Data category?").

Splitting the corpus this way keeps the knowledge base itself static and
reviewable (good for a viva -- you can literally show the source documents),
while still grounding answers in the user's personal results.
"""

from typing import List

ROLE_GUIDES = {
    "AI/ML Engineer": """
AI/ML Engineers design, build, and deploy machine learning systems that go into production, not just notebooks.
Core responsibilities include data preprocessing, model selection, training, evaluation, and deployment via APIs
or batch pipelines. Strong Python skills are non-negotiable, along with fluency in at least one deep learning
framework such as PyTorch or TensorFlow. Employers look closely at whether a candidate understands the full
lifecycle: feature engineering with Pandas/NumPy, experiment tracking, and model versioning.

MLOps knowledge -- Docker for containerizing models, basic Kubernetes for orchestration, and cloud platforms
like AWS/GCP/Azure for deployment -- increasingly separates junior candidates from mid-level ones, because it
shows the candidate can ship a model, not just train one. Interviewers commonly probe: how you handle
overfitting, how you'd debug a model that performs well offline but poorly in production, and whether you can
explain bias-variance tradeoff in plain language. Portfolio projects that show an end-to-end pipeline (data ->
training -> a deployed, callable endpoint) are far more convincing than a single Kaggle notebook.
""",
    "Data Scientist": """
Data Scientists sit at the intersection of statistics, business context, and communication. The job is less
about building the most complex model and more about answering a business question rigorously: is this
effect real, is it significant, and what should we do about it. Strong SQL is expected for almost every role,
since most work starts with pulling and shaping data, not training models.

Statistical fundamentals -- hypothesis testing, confidence intervals, experiment design (A/B testing) -- are
tested more often than deep learning theory in Data Scientist interviews. Visualization skill (matplotlib,
seaborn, or a BI tool) matters because a Data Scientist's output is usually a decision, and decisions need
clear communication. Candidates who can walk through a real analysis end-to-end -- including where the
data was messy and how they handled it -- stand out far more than those who only describe model architectures.
""",
    "Data Analyst": """
Data Analysts turn raw data into decisions for non-technical stakeholders. SQL and Excel remain the two most
requested skills for this role by a wide margin, because most day-to-day work involves querying, cleaning,
and summarizing data rather than building models. A BI tool (Tableau or Power BI) is typically expected for
building dashboards that stakeholders check regularly.

Interviewers for this role often present a messy dataset and ask the candidate to find and explain an insight,
testing both technical SQL skill and the ability to communicate findings simply. Business acumen -- understanding
what metrics actually matter to a company and why -- is frequently the differentiator between a technically
competent analyst and a highly valued one.
""",
    "Full-Stack Developer": """
Full-Stack Developers are expected to be comfortable across the whole web stack: a frontend framework (most
commonly React), a backend language/framework, and a database. REST API design is a core, heavily-tested skill,
since most full-stack roles involve building and consuming APIs daily. Git fluency, including branching and
resolving merge conflicts, is assumed at every level.

System design questions become common starting at the mid-level: how would you structure a service that needs
to scale, where would you cache, how would you handle authentication. Candidates are also often asked to justify
technology choices ("why REST over GraphQL here") rather than just describe what they used, so understanding
tradeoffs matters as much as raw tool familiarity.
""",
    "Backend Developer": """
Backend Developers own the server-side logic, APIs, and data layer that power an application. REST API design,
database schema design, and system design are the three most consistently tested areas. Strong SQL skills and
comfort with at least one relational database (PostgreSQL is common) are expected, along with growing familiarity
with caching layers like Redis for performance-sensitive systems.

At senior levels, interviews shift heavily toward system design: how to design a URL shortener, a rate limiter,
or a notification service, focusing on scalability, consistency, and failure handling. Testing discipline (unit
and integration tests) is a strong positive signal that's often specifically asked about.
""",
    "Frontend Developer": """
Frontend Developers focus on building performant, accessible user interfaces. JavaScript/TypeScript and a
modern framework (usually React) are the baseline expectation, along with solid CSS fundamentals including
responsive design. Increasingly, interviewers ask about performance (bundle size, lazy loading, rendering
optimization) and accessibility (semantic HTML, ARIA), not just component-building.

Practical coding rounds often involve building a small interactive component live -- a search-with-debounce,
a form with validation -- so speed and code cleanliness under time pressure matter as much as knowing the
framework's API surface.
""",
    "Cybersecurity Specialist": """
Cybersecurity Specialists are evaluated heavily on hands-on technical depth: network security fundamentals,
vulnerability assessment, and practical tool familiarity (Wireshark for traffic analysis, Nmap for scanning,
Metasploit for exploitation testing). Linux proficiency is close to mandatory, since most security tooling and
target environments are Linux-based.

Employers increasingly also expect awareness of compliance frameworks (ISO 27001, NIST) and cloud security
basics (IAM, security groups), since more infrastructure now lives in the cloud. Scenario-based interview
questions -- "walk me through how you'd respond to a detected intrusion" -- are common and test structured
incident-response thinking, not just tool knowledge.
""",
    "Cloud / DevOps Engineer": """
Cloud/DevOps Engineers are responsible for the infrastructure and deployment pipelines that let software ship
reliably. AWS (or another major cloud) plus Docker and Kubernetes form the core expected stack. Infrastructure-
as-Code tools like Terraform are increasingly a baseline expectation rather than a bonus, since manually
configured infrastructure doesn't scale or reproduce reliably.

CI/CD pipeline design (GitHub Actions, Jenkins, or GitLab CI) is commonly tested through scenario questions:
how would you set up a pipeline that runs tests, builds a container, and deploys safely with rollback. Strong
Linux and scripting (Bash/Python) skills underpin almost everything else in this role.
""",
    "Product Manager": """
Product Managers are evaluated on product sense, prioritization, and stakeholder communication far more than
raw technical skill. Being able to reason through prioritization frameworks (RICE, MoSCoW) and defend a
roadmap decision under pushback is a core interview skill. Agile/Scrum fluency is assumed at almost every
company.

Data literacy -- being able to read a dashboard, reason about a metric drop, and design a simple A/B test --
is increasingly expected even without deep SQL skills. The strongest candidates can walk through a product
they shipped end-to-end: the user problem, the tradeoffs considered, and how they measured success afterward.
""",
}


def get_role_guide_documents() -> List[dict]:
    """Return the static role-guide corpus as RAG-ready documents."""
    return [
        {
            "text": guide.strip(),
            "source": f"Role Guide: {role}",
            "metadata": {"type": "role_guide", "role": role},
        }
        for role, guide in ROLE_GUIDES.items()
    ]


def get_resume_documents(profile: dict, raw_resume_text: str) -> List[dict]:
    """Turn the candidate's resume + extracted profile into RAG documents."""
    docs = []
    if raw_resume_text:
        docs.append({
            "text": raw_resume_text,
            "source": "Candidate Resume (raw text)",
            "metadata": {"type": "resume_raw"},
        })
    if profile:
        summary_text = (
            f"Candidate: {profile.get('name', 'N/A')}\n"
            f"Experience Level: {profile.get('experience_level', 'N/A')}\n"
            f"Years of Experience: {profile.get('years_experience', 'N/A')}\n"
            f"Summary: {profile.get('summary', '')}\n"
            f"Skills: {', '.join(profile.get('skills', []))}\n"
            f"Education: {'; '.join(profile.get('education', []))}\n"
            f"Certifications: {'; '.join(profile.get('certifications', []))}\n"
            f"Past Roles: {'; '.join(profile.get('past_roles', []))}\n"
        )
        docs.append({
            "text": summary_text,
            "source": "Candidate Profile (structured summary)",
            "metadata": {"type": "resume_profile"},
        })
    return docs


def get_analysis_documents(gap_result: dict, roadmap: list, target_role: str) -> List[dict]:
    """Turn the candidate's own gap-analysis + roadmap results into RAG documents."""
    docs = []
    if gap_result:
        text = (
            f"Skill gap analysis for target role '{target_role}':\n"
            f"Overall match score: {gap_result.get('overall_score', 0)}%\n"
            f"Risk level: {gap_result.get('risk_level', 'N/A')}\n"
            f"Narrative: {gap_result.get('narrative', '')}\n"
            f"Matched skills: {', '.join(gap_result.get('matched_skills', []))}\n"
            f"Missing critical skills: {', '.join(gap_result.get('missing_critical_combined', []))}\n"
            f"Top priorities: {', '.join(gap_result.get('top_priorities', []))}\n"
        )
        docs.append({
            "text": text,
            "source": "Your Skill Gap Analysis Results",
            "metadata": {"type": "gap_result"},
        })
    if roadmap:
        for week in roadmap:
            text = (
                f"Week {week.get('week')}: {week.get('focus')}\n"
                f"Topics: {', '.join(week.get('topics', []))}\n"
                f"Project: {week.get('project', '')}\n"
                f"Milestone: {week.get('milestone', '')}\n"
            )
            docs.append({
                "text": text,
                "source": f"Your Roadmap - Week {week.get('week')}",
                "metadata": {"type": "roadmap_week", "week": week.get("week")},
            })
    return docs
