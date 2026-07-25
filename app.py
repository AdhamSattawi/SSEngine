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

    /* -------- Global Font & RTL (Safely target text containers without breaking icons) -------- */
    html, body, .stApp, .profile-card, .main-header, .results-header, .dev-box, .empty-state {
        font-family: 'Rubik', -apple-system, BlinkMacSystemFont, sans-serif;
        direction: rtl;
        text-align: right;
    }

    /* -------- Inputs & Selectboxes RTL -------- */
    .stTextInput input, .stSelectbox select, .stMultiSelect {
        text-align: right;
        direction: rtl;
    }

    /* -------- Main Title Header -------- */
    .main-header {
        padding: 1.25rem 0 1rem 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
        direction: rtl !important;
        text-align: right !important;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.5px;
        color: var(--text-color);
        text-align: right !important;
    }
    .main-header p {
        font-size: 0.95rem;
        margin: 0;
        opacity: 0.8;
        color: var(--text-color);
        text-align: right !important;
    }

    /* -------- Profile Card (Syncs automatically with Streamlit Theme) -------- */
    .profile-card {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .profile-card:hover {
        border-color: rgba(128, 128, 128, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
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
        margin: 0;
        color: var(--text-color);
    }
    .profile-card .card-title {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0.2rem 0 0 0;
        color: var(--text-color);
    }
    .profile-card .card-body {
        font-size: 0.88rem;
        line-height: 1.65;
        opacity: 0.9;
        color: var(--text-color);
    }

    /* -------- Score Badges -------- */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .score-badge.high {
        background-color: rgba(34, 197, 94, 0.15);
        color: var(--text-color);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .score-badge.medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: var(--text-color);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .score-badge.low {
        background-color: rgba(107, 114, 128, 0.15);
        color: var(--text-color);
        opacity: 0.8;
        border: 1px solid rgba(107, 114, 128, 0.4);
    }

    /* -------- Match Explanation Tags -------- */
    .match-tag {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.12);
        color: var(--text-color);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        margin: 0.25rem 0 0.25rem 0.35rem;
    }

    /* -------- Admin Note -------- */
    .admin-note {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        font-size: 0.82rem;
        color: var(--text-color);
        margin-top: 0.75rem;
    }

    /* -------- Similar Profiles List -------- */
    .similar-item {
        padding: 0.4rem 0;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.9;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .similar-item:last-child { border-bottom: none; }

    /* -------- Results Header -------- */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .results-header h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color);
    }
    .results-header .meta {
        font-size: 0.82rem;
        opacity: 0.75;
        color: var(--text-color);
    }

    /* -------- Empty State -------- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        opacity: 0.75;
        color: var(--text-color);
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .empty-state p { font-size: 0.95rem; margin: 0; }

    /* -------- Sidebar Polish -------- */
    section[data-testid="stSidebar"] {
        font-family: 'Rubik', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1.25rem;
        color: var(--text-color);
    }

    /* -------- Developer Mode Box -------- */
    .dev-box {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
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
    <div class="main-header" dir="rtl">
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
# Search Input & Mode Selector
# --------------------------------------------------------------------------------------
# Render search mode toggle horizontally above the search bar
mode = st.radio(
    "בחר שיטת חיפוש:",
    options=["Semantic Search", "Keyword Search"],
    horizontal=True,
    help="Semantic Search מוצא התאמות לפי הקשר ומשמעות. Keyword Search מחפש התאמה מדויקת של מילים."
)

query = st.text_input(
    "🔍 הזן שאילתת חיפוש חופשית",
    placeholder="לדוגמה: חינוך בלתי פורמלי, יזמות חברתית, מנהיגות צעירה...",
    label_visibility="collapsed"
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

    if work_df.empty:
        st.warning("לא נמצאו תוצאות המתאימות לקריטריונים שנבחרו.")
    else:
        # Result cards
        for orig_idx, row in work_df.iterrows():
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

            # "Find similar" button — Positional index lookup with get_loc
            if st.button("🔗 מצא/י אנשים דומים", key=f"similar-{orig_idx}"):
                pos_idx = df.index.get_loc(orig_idx)
                similar = find_similar(pos_idx, embeddings, df, top_k=5)
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
