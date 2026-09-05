# 🎯 Agentic AI Career Assessment & Skill Gap Analyzer

A multi-agent AI system that parses a resume, benchmarks it against real
market skill requirements for a target job role, generates a personalized
week-by-week learning roadmap and tailored mock interview questions, and
lets you chat with a **RAG + tool-calling career coach agent** that has
persistent session memory — all through an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```
career_assessment_agent/
├── app.py                    # Streamlit dashboard (UI + orchestration entry point)
├── config.py                  # Model configs, role/skill matrix, constants
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
├── agents/
│   ├── resume_agent.py          # Agent 1: Resume Parser & Skill Extractor
│   ├── gap_analyzer.py          # Agent 2: Market Gap Analyzer
│   ├── roadmap_agent.py         # Agent 3: Career Roadmap Generator
│   ├── interview_agent.py       # Agent 4: Interview Prep Generator
│   ├── chat_agent.py            # Agent 5: RAG + Tool-Calling Career Chat Agent
│   └── orchestrator.py          # LangGraph StateGraph wiring agents 1-4
└── utils/
    ├── pdf_parser.py             # PDF -> text extraction (pdfplumber + PyPDF2)
    ├── llm_client.py              # Unified Anthropic/Groq client w/ fallback + tool-calling loop
    ├── vector_store.py            # RAG core: chunking, embeddings, vector search
    ├── knowledge_base.py          # Static role-guide document corpus for RAG
    ├── tools.py                   # Custom Python functions bound to the LLM
    └── report_generator.py        # Markdown + PDF report export
```

### Multi-Agent Pipeline

```
Resume Text ──► [Resume Parser Agent] ──► structured profile (skills, education, exp.)
                                                     │
                                                     ▼
                                        [Market Gap Analyzer Agent]
                                     (deterministic scoring + LLM narrative)
                                                     │
                                        ┌────────────┴────────────┐
                                        ▼                          ▼
                          [Career Roadmap Agent]        [Interview Prep Agent]
                       (week-by-week plan, resources)   (weighted mock questions)
                                        │                          │
                                        └────────────┬─────────────┘
                                                      ▼
                                    [RAG + Tool-Calling Chat Agent]
                     Retrieves grounded context from a vector store (role guides +
                     resume + your results) AND can call live tools (skill-gap
                     lookup, roadmap lookup, interview-question search) — the LLM
                     decides for itself when to retrieve vs. when to call a tool.
                     Full conversation history is remembered across turns.
```

Each agent can run **independently** from its own tab in the UI, or agents
1–4 can be chained together via the `agents/orchestrator.py` LangGraph
pipeline. The Chat Agent (tab 5) is always conversational/on-demand.

### RAG Pipeline (Tab 5: AI Career Chat)

```
Documents (role guides + your resume + your analysis results)
        │
        ▼
  Chunking            RecursiveCharacterTextSplitter (chunk_size=500, overlap=80)
        │              -- falls back to a dependency-free splitter if unavailable
        ▼
  Embeddings          sentence-transformers (all-MiniLM-L6-v2), local, no API key
        │              -- falls back to a numpy TF-IDF vectorizer if unavailable
        ▼
  Vector Store        FAISS in-memory index (cosine similarity via inner product)
        │              -- falls back to plain numpy cosine similarity if unavailable
        ▼
  Retrieval           top-k most relevant chunks returned via similarity_search()
        │
        ▼
  Exposed as a TOOL   search_career_knowledge() -- the LLM calls this itself
                       when it decides a question needs grounded context
```

### Tool Binding & Function Calling (Tab 5)

The chat agent binds 4 custom Python functions to the LLM via native
tool-calling (Anthropic's `tools` param / Groq's OpenAI-compatible `tools`
param) in `utils/tools.py`:

| Tool | Purpose |
|---|---|
| `search_career_knowledge` | RAG retrieval over role guides + resume + results |
| `get_skill_gap_summary` | Returns the candidate's precomputed gap analysis |
| `get_roadmap_week` | Returns a specific week's learning plan |
| `get_interview_questions_by_topic` | Filters generated interview questions |

The LLM decides **for itself**, per user message, whether to call zero, one,
or several of these tools before composing its final answer — see
`utils/llm_client.py`'s `chat_with_tools()` for the full agentic loop
(handles both Anthropic's and Groq's tool-calling formats).

### Reliability Design

- **Deterministic scoring, LLM narrative**: skill-gap match percentages are
  computed with a static, reproducible skill-weight matrix + fuzzy string
  matching — never solely by the LLM. The LLM only adds qualitative
  narrative on top, so scores are consistent and explainable.
- **Automatic provider fallback**: if your primary provider (Anthropic or
  Groq) fails or hits a rate limit, the app automatically retries with the
  other provider if a key is available.
- **Graceful degradation**: every agent has a deterministic, rule-based
  fallback path if the LLM call fails entirely (bad key, network issue),
  so the app never hard-crashes.
- **RAG degrades gracefully too**: if `sentence-transformers`/`faiss-cpu`
  aren't installed, `utils/vector_store.py` automatically falls back to a
  dependency-free numpy TF-IDF index, so the chat tab still works (with
  slightly less semantic retrieval quality) even in a minimal environment.

---


## 🚀 Quick Start (Local)

### 1. Clone / download this project

```bash
cd career_assessment_agent
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (optional but recommended)

You can either enter keys directly in the sidebar at runtime, **or** create a
`.env` file in the project root so they're pre-filled:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
GROQ_API_KEY=gsk-your-key-here
```

You only need **one** of the two keys to run the app — Anthropic Claude
(`claude-3-5-sonnet` / `claude-3-haiku`) or Groq (`llama-3.3-70b-versatile`).
Providing both enables automatic fallback if one provider is unavailable.

- Get an Anthropic key: https://console.anthropic.com/
- Get a Groq key: https://console.groq.com/keys

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧭 Using the App

1. **Sidebar** — enter your API key(s), choose your target job role, your
   experience level, and roadmap/interview preferences.
2. **Tab 1 (📄 Resume Upload)** — upload a PDF resume or paste text, then
   click **Analyze Resume** to see your extracted skills, education, and
   experience profile.
3. **Tab 2 (🎯 Skill Gap Analysis)** — click **Run Gap Analysis** to see a
   radar chart and detailed skill matrix comparing you to the target role.
4. **Tab 3 (🗺️ Career Roadmap)** — click **Generate Roadmap** to get a
   week-by-week learning plan with resources, projects, and milestones.
5. **Tab 4 (💡 Interview Prep)** — click **Generate Interview Questions**
   for tailored mock interview questions weighted toward your weak areas.
6. **Tab 5 (💬 AI Career Chat)** — chat with Aria, the RAG + tool-calling
   career coach. Ask things like *"Why is Docker important for my target
   role?"* (triggers RAG retrieval) or *"What's my current match score?"*
   (triggers the `get_skill_gap_summary` tool) or *"What should I study in
   week 3?"* (triggers `get_roadmap_week`). Expand the **🔧 tool call(s)**
   note under any answer to see exactly which tool(s) fired and what they
   returned — useful for demonstrating tool-calling live in a viva. The full
   conversation is remembered for the session; use **Clear Chat History** to
   reset it.
6. **Export** — scroll to the bottom to download the full report as
   **Markdown** or **PDF**.

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (easiest)

1. Push this project to a public (or private) GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click **New app**, select your repo, branch, and set the main file path
   to `app.py`.
4. (Optional) Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GROQ_API_KEY = "gsk-..."
   ```
   These will be picked up automatically by `python-dotenv` / `os.getenv`
   as default values — users can still override them in the sidebar.
5. Click **Deploy**. Your app will be live at
   `https://<your-app-name>.streamlit.app`.

### Option B — Render

1. Push this project to a GitHub repository.
2. In Render, click **New → Web Service** and connect your repo.
3. Configure the service:
   - **Environment**: Python 3
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
4. Under **Environment**, add `ANTHROPIC_API_KEY` and/or `GROQ_API_KEY` as
   environment variables (optional — users can also enter keys in the UI).
5. Click **Create Web Service**. Render will build and deploy automatically.

### Option C — Docker (any cloud)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t career-analyzer .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... career-analyzer
```

---

## 🛠️ Extending the System

- **Add a new target role**: add an entry to `ROLE_SKILL_MATRIX` in
  `config.py` with categories, skills, and importance weights (1–3).
- **Swap/add an LLM provider**: extend `utils/llm_client.py`'s `LLMClient`
  with a new `_call_<provider>()` method and add it to `provider_order`.
- **Change roadmap length or interview question count defaults**: edit
  `ROADMAP_DEFAULT_WEEKS` / `INTERVIEW_DEFAULT_QUESTIONS` in `config.py`.
- **Use the full LangGraph pipeline in one click**: call
  `agents.orchestrator.run_pipeline(...)` from `app.py` if you want a
  single "Run Full Analysis" button instead of per-tab actions.

---

## 🔒 Privacy Note

Resume text and analysis results are stored only in the browser's Streamlit
session state for the duration of your session — nothing is persisted to a
database or file on the server.

---

## 📄 License

MIT — use freely for personal, educational, or commercial projects.
