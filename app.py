# -*- coding: utf-8 -*-
"""
Maoz — Semantic Search POC
מנוע חיפוש סמנטי לרשת מנהיגי מעוז — שכבת ממשק (Streamlit UI)

הרצה:
    pip install streamlit pandas numpy sentence-transformers
    streamlit run app.py
"""

import time
import streamlit as st

from engine import (
    file_hash_of,
    load_data,
    load_model,
    embed_corpus,
    embed_query,
    semantic_search,
    keyword_search,
    explain_match,
    find_similar,
    apply_rbac_filter,
    apply_metadata_filters,
    score_level,
)

# --------------------------------------------------------------------------------------
# הגדרות עמוד + RTL + עיצוב
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="מעוז — מנוע חיפוש סמנטי",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* -------- Google Font -------- */
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');

    /* -------- Global RTL & Typography -------- */
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'Rubik', sans-serif;
    }

    /* -------- Inputs -------- */
    .stTextInput input, .stSelectbox select, .stMultiSelect {
        text-align: right;
        direction: rtl;
        font-family: 'Rubik', sans-serif;
    }

    /* -------- Main title area -------- */
    .main-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid rgba(49, 51, 63, 0.1);
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }

    /* -------- Profile card -------- */
    .profile-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .profile-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        border-color: #d1d5db;
    }
    .profile-card .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .profile-card .card-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
    }
    .profile-card .card-title {
        font-size: 0.9rem;
        color: #6b7280;
        margin: 0.15rem 0 0 0;
    }
    .profile-card .card-body {
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.65;
    }

    /* -------- Score badge -------- */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .score-badge.high {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .score-badge.medium {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .score-badge.low {
        background: #f9fafb;
        color: #6b7280;
        border: 1px solid #e5e7eb;
    }

    /* -------- Match explanation tags -------- */
    .match-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
        margin: 0.15rem 0 0.15rem 0.35rem;
    }

    /* -------- Admin note -------- */
    .admin-note {
        background: #fef3c7;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-top: 0.75rem;
    }

    /* -------- Similar profiles list -------- */
    .similar-item {
        padding: 0.35rem 0;
        font-size: 0.85rem;
        color: #374151;
        border-bottom: 1px solid #f3f4f6;
    }
    .similar-item:last-child { border-bottom: none; }

    /* -------- Results header -------- */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .results-header h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
    }
    .results-header .meta {
        font-size: 0.82rem;
        color: #9ca3af;
    }

    /* -------- Empty state -------- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #9ca3af;
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .empty-state p { font-size: 0.95rem; margin: 0; }

    /* -------- Sidebar polish -------- */
    section[data-testid="stSidebar"] {
        font-family: 'Rubik', sans-serif;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1.25rem;
    }

    /* -------- Dev mode box -------- */
    .dev-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Cached loaders (Streamlit caching wraps the engine functions)
# --------------------------------------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data(csv_path: str, file_hash: str):
    return load_data(csv_path)


@st.cache_data
def get_embeddings(_model, texts: tuple, file_hash: str):
    return embed_corpus(_model, list(texts))


# --------------------------------------------------------------------------------------
# UI — Header
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 מנוע חיפוש חכם לרשת מנהיגי מעוז</h1>
        <p>Proof of Concept — חיפוש לפי ניסיון, תחומי עניין, ערכים ואתגרים משותפים</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ הגדרות")
csv_path = st.sidebar.text_input("נתיב קובץ CSV", value="maoz_profiles.csv")

fhash = file_hash_of(csv_path)
try:
    df = get_data(csv_path, fhash)
except Exception as e:
    st.error(f"שגיאה בטעינת הקובץ '{csv_path}': {e}")
    st.stop()

model = get_model()
embeddings = get_embeddings(model, tuple(df["search_text"].tolist()), fhash)

st.sidebar.success(f"✓ נטענו {len(df)} פרופילים")

st.sidebar.divider()
st.sidebar.markdown("## 🔐 הרשאות")
role = st.sidebar.selectbox("תפקיד המשתמש", ["חבר/ת רשת", "צוות מעוז (Admin)"])

st.sidebar.divider()
st.sidebar.markdown("## 🔎 שיטת חיפוש")
mode = st.sidebar.radio("בחר/י מנוע", ["Semantic Search", "Keyword Search"])

st.sidebar.divider()
st.sidebar.markdown("## 📂 סינון")
region_options = sorted([r for r in df["region"].unique() if r])
sector_options = sorted([s for s in df["sector"].unique() if s])
region_filter = st.sidebar.multiselect("מחוז", region_options)
sector_filter = st.sidebar.multiselect("מגזר", sector_options)

st.sidebar.divider()
dev_mode = st.sidebar.checkbox("🛠 Developer Mode")

if "latencies" not in st.session_state:
    st.session_state["latencies"] = []

st.sidebar.divider()
st.sidebar.markdown("## 📊 Dashboard")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("פרופילים", f"{len(df):,}")
col_b.metric("ממד הטמעה", embeddings.shape[1])
st.sidebar.caption("מודל: MiniLM multilingual")
if st.session_state["latencies"]:
    avg_latency = sum(st.session_state["latencies"]) / len(st.session_state["latencies"])
    st.sidebar.caption(f"⏱ זמן חיפוש ממוצע: {avg_latency*1000:.0f} ms")

# --------------------------------------------------------------------------------------
# Search Input
# --------------------------------------------------------------------------------------
query = st.text_input(
    "🔍 הזן שאילתת חיפוש חופשית",
    placeholder="לדוגמה: חינוך בלתי פורמלי, יזמות חברתית, מנהיגות צעירה...",
)

if query:
    t0 = time.time()

    if mode == "Semantic Search":
        query_vec = embed_query(model, query)
        scores = semantic_search(query_vec, embeddings)
    else:
        scores = keyword_search(query, df)

    elapsed = time.time() - t0
    st.session_state["latencies"].append(elapsed)

    work_df = df.copy()
    work_df["score"] = scores

    # RBAC + metadata filtering
    work_df = apply_rbac_filter(work_df, role)
    work_df = apply_metadata_filters(work_df, region_filter, sector_filter)

    work_df = work_df[work_df["score"] > 0].sort_values("score", ascending=False).head(10)

    # Results header
    st.markdown(
        f"""
        <div class="results-header">
            <h3>תוצאות ({len(work_df)})</h3>
            <span class="meta">{mode} · {elapsed*1000:.0f} ms</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Developer mode panel
    if dev_mode:
        st.markdown(
            f"""
            <div class="dev-box">
                Query: '{query}'<br>
                → Embedding generated: {mode == 'Semantic Search'}<br>
                → Scoring method: {'cosine similarity' if mode == 'Semantic Search' else 'keyword overlap'}<br>
                → Candidates before RBAC filter: {len(df)}<br>
                → Candidates after RBAC filter: {len(work_df)}<br>
                → Search time: {elapsed*1000:.1f} ms
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Result cards
    for idx, row in work_df.iterrows():
        score_pct = row["score"] * 100
        level = score_level(row["score"])

        name = row["name"]
        title = row["title"]
        org = row["organization"]
        title_line = f"{title} · {org}" if org else title

        experience = str(row.get("experience", ""))
        exp_snippet = experience[:250] + "..." if len(experience) > 250 else experience

        # Match explanation tags
        match_tags_html = ""
        if mode == "Semantic Search":
            labels = explain_match(query, row)
            tags = "".join(f'<span class="match-tag">{label}</span>' for label in labels)
            match_tags_html = f'<div style="margin-top: 0.5rem;">{tags}</div>'

        # Admin note
        admin_html = ""
        if role == "צוות מעוז (Admin)" and row.get("internal_note"):
            admin_html = f'<div class="admin-note">🔒 {row["internal_note"]}</div>'

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="card-header">
                    <div>
                        <p class="card-name">👤 {name}</p>
                        <p class="card-title">{title_line}</p>
                    </div>
                    <span class="score-badge {level}">{score_pct:.1f}%</span>
                </div>
                <div class="card-body">{exp_snippet}</div>
                {match_tags_html}
                {admin_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # "Find similar" button
        if st.button("🔗 מצא/י אנשים דומים", key=f"similar-{idx}"):
            similar = find_similar(idx, embeddings, df, top_k=5)
            st.markdown("**פרופילים דומים:**")
            for _, srow in similar.iterrows():
                sim_name = srow["name"]
                sim_title = srow["title"]
                sim_org = srow["organization"]
                sim_label = f"{sim_name} · {sim_title}"
                if sim_org:
                    sim_label += f" · {sim_org}"
                st.markdown(
                    f'<div class="similar-item">↳ {sim_label} — {srow["score"]*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🔍</div>
            <p>הזן שאילתה בשדה למעלה כדי להתחיל בחיפוש</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
