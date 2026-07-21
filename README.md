# 🔍 Maoz Semantic Search Engine

**מנוע חיפוש סמנטי לרשת מנהיגי מעוז**

A semantic search engine POC for the [Maoz](https://www.maoz.org.il/) leadership network, enabling intelligent profile matching based on experience, interests, values, and shared challenges — beyond simple keyword overlap.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

- **Semantic Search** — Finds conceptually related profiles even without exact keyword matches (e.g., searching "תנועות נוער" surfaces profiles mentioning "חינוך בלתי פורמלי")
- **Keyword Search** — Naive token-overlap baseline for comparison, demonstrating the value of semantic understanding
- **RBAC (Role-Based Access Control)** — Pre-filters sensitive profiles before returning results; admin-only fields are hidden from regular users
- **Find Similar People** — Given any profile, discovers the most similar leaders in vector space
- **Match Explanation** — Rule-based explainability showing *why* a profile was matched
- **Developer Mode** — Pipeline transparency panel showing query processing, scoring method, and latency
- **RTL Hebrew UI** — Fully right-to-left interface with professional typography (Google Fonts — Rubik)

---

## 🏗️ Architecture

```
[ CSV Data ] ──► [ Preprocessor ] ──► [ Embedding Model ] ──► [ In-Memory Vector Index ]
                                         (multilingual)              (NumPy)
                                                                        │
[ User Query ] ──► [ RBAC Filter ] ──► [ Similarity Search ] ──► [ Streamlit UI ]
```

| Component | Choice | Rationale |
|---|---|---|
| **Embedding Model** | `paraphrase-multilingual-MiniLM-L12-v2` | Lightweight, multilingual (Hebrew), runs locally — no paid API needed |
| **Vector Storage** | In-memory NumPy | ~1,200 profiles don't justify a dedicated vector DB; sub-millisecond search times |
| **UI Framework** | Streamlit | Rapid prototyping, easy deployment, built-in caching |
| **Data Source** | CSV export | Simulates Salesforce export for POC phase |

---

## 📁 Project Structure

```
SSEngine/
├── app.py              # Streamlit UI layer
├── engine.py           # Core search engine (no Streamlit dependency)
├── maoz_profiles.csv   # Profile data (1,206 leaders)
├── requirements.txt    # Python dependencies
├── run.bat             # Windows auto-run script (CMD)
├── run.ps1             # Windows auto-run script (PowerShell)
└── README.md
```

---

## 🚀 Quick Start

### Option A — One-Click (Windows)

Double-click **`run.bat`** — it will create a virtual environment, install dependencies, and launch the app.

### Option B — Manual Setup

```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🔎 Usage

1. **Enter a search query** in natural language (Hebrew or English)
2. **Choose search mode** — Semantic Search or Keyword Search (sidebar)
3. **Set user role** — "חבר/ת רשת" (member) or "צוות מעוז (Admin)" to see RBAC in action
4. **Explore results** — Each profile card shows name, title, background snippet, match score, and explanation
5. **Find similar people** — Click the button on any result to discover related profiles

### Example Queries

| Query | What it finds |
|---|---|
| `יזמות חברתית` | Profiles related to social entrepreneurship |
| `חינוך בלתי פורמלי` | Informal education leaders (also matches "תנועות נוער", "מועדוניות") |
| `מנהיגות צעירה` | Young leadership programs and mentors |
| `טכנולוגיה וחדשנות` | Tech & innovation leaders across sectors |

---

## 🔐 Security & Privacy

- **Pre-filtering, not post-filtering** — RBAC checks happen before results are returned; unauthorized data is never exposed
- **Local model execution** — The embedding model runs locally; no data is sent to external APIs
- **Minimal sensitive data in embeddings** — Identifying/sensitive fields are excluded from the text used for embedding generation
- **Demo RBAC fields** — Since the data was scraped from public pages, sensitive fields (`is_sensitive`, `internal_note`) are simulated for demonstration purposes only

---

## 🗺️ Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1 — POC** | ✅ Done | CSV import, local embeddings, in-memory search, Streamlit UI |
| **Phase 2 — Production** | 🔜 Planned | Direct Salesforce API integration, batch sync, hybrid search (semantic + keyword) |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive web UI
- **sentence-transformers** — Multilingual embedding model
- **pandas** — Data loading & manipulation
- **NumPy** — Vector similarity computation

---

## 📄 License

MIT
