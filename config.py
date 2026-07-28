"""
config.py
Central configuration for the Agentic AI Career Assessment & Skill Gap Analyzer.

Holds:
- Environment / API key resolution
- Model defaults for Anthropic and Groq
- Static role -> required skill matrix used by the Gap Analyzer Agent
- Small shared constants (experience levels, roadmap defaults, etc.)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# API / Model configuration
# --------------------------------------------------------------------------

# Environment variables are used as a fallback if the user does not type
# a key into the Streamlit sidebar.
DEFAULT_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Anthropic Claude models
ANTHROPIC_MODEL_SONNET = "claude-3-5-sonnet-20241022"
ANTHROPIC_MODEL_HAIKU = "claude-3-haiku-20240307"
DEFAULT_ANTHROPIC_MODEL = ANTHROPIC_MODEL_SONNET

# Groq models
GROQ_MODEL_LLAMA_70B = "llama-3.3-70b-versatile"
DEFAULT_GROQ_MODEL = GROQ_MODEL_LLAMA_70B

# Order in which providers are attempted when "auto" fallback mode is used.
DEFAULT_PROVIDER_ORDER = ["anthropic", "groq"]

# Generation defaults
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 2500

# --------------------------------------------------------------------------
# App-level constants
# --------------------------------------------------------------------------

APP_TITLE = "🎯 Agentic AI Career Assessment & Skill Gap Analyzer"
APP_ICON = "🎯"

EXPERIENCE_LEVELS = ["Student / Fresher", "Entry-Level (0-2 yrs)", "Mid-Level (2-5 yrs)",
                     "Senior (5-8 yrs)", "Lead / Principal (8+ yrs)"]

TARGET_ROLES = [
    "AI/ML Engineer",
    "Data Scientist",
    "Data Analyst",
    "Full-Stack Developer",
    "Backend Developer",
    "Frontend Developer",
    "Cybersecurity Specialist",
    "Cloud / DevOps Engineer",
    "Product Manager",
]

ROADMAP_DEFAULT_WEEKS = 8
INTERVIEW_DEFAULT_QUESTIONS = 8

# --------------------------------------------------------------------------
# Role -> Required Skill Matrix
# --------------------------------------------------------------------------
# Structure:
# {
#   "Role Name": {
#       "Category Name": {
#            "Skill Name": weight (1 = nice-to-have, 2 = important, 3 = critical)
#       }
#   }
# }
# This matrix acts as the ground-truth "market requirement" baseline that the
# Market Gap Analyzer Agent compares extracted resume skills against. The LLM
# is used on top of this deterministic matrix to add qualitative narrative,
# so the numeric score is always reproducible even if the LLM call fails.

ROLE_SKILL_MATRIX = {
    "AI/ML Engineer": {
        "Programming": {"Python": 3, "SQL": 2, "C++": 1},
        "ML/DL Frameworks": {"PyTorch": 3, "TensorFlow": 2, "Scikit-learn": 3, "Keras": 1},
        "Core Concepts": {"Machine Learning": 3, "Deep Learning": 3, "NLP": 2,
                           "Computer Vision": 1, "Statistics": 2, "Linear Algebra": 2},
        "Data Engineering": {"Pandas": 3, "NumPy": 3, "Feature Engineering": 2, "Data Preprocessing": 2},
        "MLOps & Tools": {"Docker": 2, "Git": 3, "MLflow": 1, "Kubernetes": 1,
                           "AWS": 1, "GCP": 1, "Azure": 1},
    },
    "Data Scientist": {
        "Programming": {"Python": 3, "R": 1, "SQL": 3},
        "Statistics & Math": {"Statistics": 3, "Probability": 2, "Hypothesis Testing": 2, "Linear Algebra": 1},
        "ML Frameworks": {"Scikit-learn": 3, "PyTorch": 1, "TensorFlow": 1, "XGBoost": 2},
        "Data Tools": {"Pandas": 3, "NumPy": 3, "Data Visualization": 3, "Excel": 1},
        "Visualization & BI": {"Tableau": 1, "Power BI": 1, "Matplotlib": 2, "Seaborn": 2},
    },
    "Data Analyst": {
        "Programming": {"SQL": 3, "Python": 2, "Excel": 3},
        "Analysis": {"Statistics": 2, "Data Cleaning": 3, "Data Visualization": 3},
        "BI Tools": {"Tableau": 2, "Power BI": 2, "Looker": 1},
        "Reporting": {"Dashboarding": 2, "A/B Testing": 1, "Business Acumen": 2},
    },
    "Full-Stack Developer": {
        "Frontend": {"JavaScript": 3, "React": 3, "HTML": 2, "CSS": 2, "TypeScript": 2},
        "Backend": {"Node.js": 2, "Python": 2, "REST APIs": 3, "GraphQL": 1},
        "Database": {"SQL": 3, "MongoDB": 1, "PostgreSQL": 2},
        "DevOps & Tools": {"Git": 3, "Docker": 2, "CI/CD": 2, "AWS": 1},
        "System Design": {"System Design": 2, "Microservices": 1, "Testing": 2},
    },
    "Backend Developer": {
        "Languages": {"Python": 2, "Java": 2, "Node.js": 2, "Go": 1},
        "APIs & Architecture": {"REST APIs": 3, "Microservices": 2, "System Design": 3, "GraphQL": 1},
        "Database": {"SQL": 3, "PostgreSQL": 2, "MongoDB": 1, "Redis": 1},
        "DevOps": {"Docker": 2, "Kubernetes": 1, "CI/CD": 2, "Git": 3},
        "Testing & Quality": {"Unit Testing": 2, "Integration Testing": 1},
    },
    "Frontend Developer": {
        "Core": {"JavaScript": 3, "HTML": 3, "CSS": 3, "TypeScript": 2},
        "Frameworks": {"React": 3, "Vue": 1, "Angular": 1, "Next.js": 2},
        "Styling": {"Tailwind CSS": 2, "Responsive Design": 2, "SASS": 1},
        "Tooling": {"Git": 3, "Webpack": 1, "Testing": 2, "Accessibility": 1},
    },
    "Cybersecurity Specialist": {
        "Core Security": {"Network Security": 3, "Cryptography": 2, "Threat Modeling": 2,
                           "Vulnerability Assessment": 3},
        "Tools": {"Wireshark": 2, "Nmap": 2, "Metasploit": 2, "SIEM Tools": 2, "Burp Suite": 1},
        "Compliance & Governance": {"Risk Assessment": 2, "ISO 27001": 1, "GDPR": 1, "NIST Framework": 1},
        "Technical": {"Linux": 3, "Python": 2, "Penetration Testing": 3, "Incident Response": 2},
        "Cloud Security": {"AWS Security": 1, "Azure Security": 1, "IAM": 2},
    },
    "Cloud / DevOps Engineer": {
        "Cloud Platforms": {"AWS": 3, "Azure": 1, "GCP": 1},
        "Containers & Orchestration": {"Docker": 3, "Kubernetes": 3, "Helm": 1},
        "IaC & Automation": {"Terraform": 2, "Ansible": 1, "CloudFormation": 1},
        "CI/CD": {"Jenkins": 2, "GitHub Actions": 2, "CI/CD": 3, "GitLab CI": 1},
        "Scripting & Monitoring": {"Python": 2, "Bash": 2, "Prometheus": 1, "Grafana": 1, "Linux": 3},
    },
    "Product Manager": {
        "Strategy": {"Product Strategy": 3, "Market Research": 2, "Roadmapping": 3},
        "Execution": {"Agile/Scrum": 3, "User Stories": 2, "Prioritization Frameworks": 2},
        "Analytics": {"Data Analysis": 2, "SQL": 1, "A/B Testing": 2, "Product Analytics": 2},
        "Collaboration": {"Stakeholder Management": 3, "Communication": 3, "UX Fundamentals": 1},
    },
}


def get_role_skill_matrix(role: str) -> dict:
    """Return the required skill matrix for a role, or an empty dict if unknown."""
    return ROLE_SKILL_MATRIX.get(role, {})


def get_all_role_skills_flat(role: str) -> dict:
    """Flatten the category->skill->weight matrix into a single skill->weight dict."""
    flat = {}
    for category, skills in ROLE_SKILL_MATRIX.get(role, {}).items():
        flat.update(skills)
    return flat
