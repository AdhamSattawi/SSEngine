# -*- coding: utf-8 -*-
"""
Maoz — Semantic Search POC
מנוע חיפוש סמנטי לרשת מנהיגי מעוז

הרצה:
    pip install streamlit pandas numpy sentence-transformers
    streamlit run app.py

קלט:
    קובץ CSV בשם maoz_profiles.csv באותה תיקייה (ברירת מחדל, ניתן לשנות בסרגל הצד).
    הקוד מנרמל שמות עמודות נפוצים - ראו COLUMN_ALIASES למטה.
    אם לקובץ שלך יש שמות עמודות שונים, פשוט הוסף אותם למילון המיפוי.

הערה לגבי is_sensitive / internal_note:
    הנתונים נאספו מהעמודים הציבוריים באתר מעוז (Web Scraping) ולכן אינם כוללים
    שדות פרטיים אמיתיים (טלפון, הערות פנימיות וכו'). כדי להדגים את מנגנון
    ההרשאות (RBAC) המבוקש בתרגיל, אם העמודות is_sensitive / internal_note
    לא קיימות בקובץ - הקוד מדמה אותן (בצורה מבוקרת ומתועדת, ולא כמידע אמיתי).
"""

import time
import re
import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------------------
# הגדרות עמוד + RTL
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="מעוז — מנוע חיפוש סמנטי", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; text-align: right; }
    .stTextInput input { text-align: right; direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# מיפוי שמות עמודות אפשריים בקובץ הגלם -> שם קנוני בקוד.
# הוסיפו כאן כל שם עמודה נוסף שקיים בקובץ ה-CSV שלכם.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "name": ["name", "full_name", "שם", "שם מלא"],
    "title": ["title", "role", "position", "תפקיד"],
    "organization": ["organization", "org", "company", "ארגון"],
    "region": ["region", "district", "מחוז", "אזור"],
    "sector": ["sector", "מגזר"],
    "experience": ["experience", "bio", "about", "ניסיון", "תקציר"],
    "interests": ["interests", "tags", "תחומי עניין"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ממפה עמודות מהקובץ הגולמי לשמות הקנוניים שהאפליקציה מצפה להם."""
    rename_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                rename_map[lower_cols[alias.lower()]] = canonical
                break
    df = df.rename(columns=rename_map)
    for canonical in COLUMN_ALIASES:
        if canonical not in df.columns:
            df[canonical] = ""
    return df


def add_demo_rbac_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    הנתונים נאספו מעמודים ציבוריים ולכן אין בהם שדות רגישים אמיתיים.
    כדי להדגים את מנגנון ה-RBAC בלבד (לא כדי לייצג מידע אמיתי), מסמנים
    חלק מהפרופילים כ'רגישים' באופן דטרמיניסטי (seed קבוע) ומוסיפים הערה
    מדומה שגלויה רק לתפקיד Admin.
    """
    if "is_sensitive" not in df.columns:
        rng = np.random.RandomState(42)
        df["is_sensitive"] = rng.rand(len(df)) < 0.15
    if "internal_note" not in df.columns:
        df["internal_note"] = np.where(
            df["is_sensitive"],
            "הערה פנימית לדוגמה (מידע מדומה לצורך הדגמת RBAC בלבד)",
            "",
        )
    return df


def build_search_text(row: pd.Series) -> str:
    return (
        f"שם: {row.get('name', '')}. "
        f"תפקיד: {row.get('title', '')}. "
        f"ארגון: {row.get('organization', '')}. "
        f"מחוז: {row.get('region', '')}. "
        f"מגזר: {row.get('sector', '')}. "
        f"ניסיון: {row.get('experience', '')}. "
        f"תחומי עניין: {row.get('interests', '')}."
    )


@st.cache_resource
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_data(csv_path: str, file_hash: str):
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    df = add_demo_rbac_fields(df)
    df["search_text"] = df.apply(build_search_text, axis=1)
    return df


@st.cache_data
def embed_corpus(_model: SentenceTransformer, texts: tuple, file_hash: str):
    embeddings = _model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def keyword_search(query: str, df: pd.DataFrame) -> np.ndarray:
    """
    בסיס להשוואה: חיפוש מילות מפתח נאיבי (לא סמנטי).
    ציון = יחס הטוקנים בשאילתה שמופיעים כמחרוזת-משנה בטקסט הפרופיל.
    במכוון פשוט - המטרה היא להראות את הפער מול חיפוש סמנטי, לא לבנות BM25 מלא.
    """
    tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    if not tokens:
        return np.zeros(len(df))
    scores = np.zeros(len(df))
    texts_lower = df["search_text"].str.lower().tolist()
    for i, text in enumerate(texts_lower):
        hits = sum(1 for t in tokens if t.lower() in text)
        scores[i] = hits / len(tokens)
    return scores


def explain_match(query: str, row: pd.Series) -> List[str]:
    """
    הסבר מבוסס-חוקים (לא LLM) - בודק אילו שדות בפרופיל חולקים מילים עם השאילתה.
    הערה: זהו היוריסטיקה פשוטה שמזהה חפיפת מילים מדויקת; היא *לא* "רואה"
    קשרים סמנטיים כמו "חינוך בלתי פורמלי" מול "תנועות נוער" - את זה עושה
    שכבת ה-Embeddings. אם אין חפיפת מילים ישירה, הפרופיל עדיין יכול להיות
    התאמה סמנטית טובה, ואז מוצג הסבר כללי.
    """
    tokens = set(t.lower() for t in re.split(r"\s+", query.strip()) if t)
    field_labels = {"experience": "ניסיון", "interests": "תחומי עניין", "title": "תפקיד"}
    matched = []
    for field, label in field_labels.items():
        value = str(row.get(field, "")).lower()
        if any(t in value for t in tokens):
            matched.append(label)
    if not matched:
        matched.append("התאמה סמנטית כללית (ללא חפיפת מילים ישירה)")
    return matched


def find_similar(idx: int, embeddings: np.ndarray, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    target = embeddings[idx]
    scores = np.dot(embeddings, target)
    order = np.argsort(-scores)
    order = [i for i in order if i != idx][:top_k]
    result = df.iloc[order].copy()
    result["score"] = scores[order]
    return result


def file_hash_of(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "missing"


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("🔍 מנוע חיפוש חכם לרשת מנהיגי מעוז")
st.caption("Proof of Concept — חיפוש לפי ניסיון, תחומי עניין, ערכים ואתגרים משותפים")

st.sidebar.header("נתונים")
csv_path = st.sidebar.text_input("נתיב קובץ CSV", value="maoz_profiles.csv")

fhash = file_hash_of(csv_path)
try:
    df = load_data(csv_path, fhash)
except Exception as e:
    st.error(f"שגיאה בטעינת הקובץ '{csv_path}': {e}")
    st.stop()

model = get_model()
embeddings = embed_corpus(model, tuple(df["search_text"].tolist()), fhash)

st.sidebar.success(f"נטענו {len(df)} פרופילים")

st.sidebar.header("הרשאות")
role = st.sidebar.selectbox("תפקיד המשתמש", ["חבר/ת רשת", "צוות מעוז (Admin)"])

st.sidebar.header("שיטת חיפוש")
mode = st.sidebar.radio("בחר/י מנוע", ["Semantic Search", "Keyword Search"])

st.sidebar.header("סינון")
region_options = sorted([r for r in df["region"].unique() if r])
sector_options = sorted([s for s in df["sector"].unique() if s])
region_filter = st.sidebar.multiselect("מחוז", region_options)
sector_filter = st.sidebar.multiselect("מגזר", sector_options)

dev_mode = st.sidebar.checkbox("Developer Mode")

if "latencies" not in st.session_state:
    st.session_state["latencies"] = []

st.sidebar.header("📊 Dashboard")
st.sidebar.metric("Profiles", len(df))
st.sidebar.metric("Embedding Model", "MiniLM multilingual")
st.sidebar.metric("Embedding Dim", embeddings.shape[1])
if st.session_state["latencies"]:
    avg_latency = sum(st.session_state["latencies"]) / len(st.session_state["latencies"])
    st.sidebar.metric("Avg Search Time", f"{avg_latency*1000:.0f} ms")

query = st.text_input(
    "הזן שאילתת חיפוש חופשית",
    placeholder="לדוגמה: חינוך בלתי פורמלי, יזמות חברתית, מנהיגות צעירה...",
)

if query:
    t0 = time.time()

    if mode == "Semantic Search":
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(embeddings, query_vec)
    else:
        scores = keyword_search(query, df)

    elapsed = time.time() - t0
    st.session_state["latencies"].append(elapsed)

    work_df = df.copy()
    work_df["score"] = scores

    # RBAC pre-filtering — מתבצע לפני הצגת תוצאות, לא אחרי
    if role == "חבר/ת רשת":
        work_df = work_df[work_df["is_sensitive"] != True]  # noqa: E712

    if region_filter:
        work_df = work_df[work_df["region"].isin(region_filter)]
    if sector_filter:
        work_df = work_df[work_df["sector"].isin(sector_filter)]

    work_df = work_df[work_df["score"] > 0].sort_values("score", ascending=False).head(10)

    st.subheader(f"תוצאות ({len(work_df)}) — {mode}")

    if dev_mode:
        with st.expander("🛠 Developer Mode — Pipeline", expanded=True):
            st.code(
                f"Query: '{query}'\n"
                f"→ Embedding generated: {mode == 'Semantic Search'}\n"
                f"→ Scoring method: {'cosine similarity' if mode == 'Semantic Search' else 'keyword overlap'}\n"
                f"→ Candidates before RBAC filter: {len(df)}\n"
                f"→ Candidates after RBAC filter: {len(work_df)}\n"
                f"→ Search time: {elapsed*1000:.1f} ms",
                language="text",
            )

    for idx, row in work_df.iterrows():
        header = f"👤 {row['name']} | {row['title']} ({row['organization']}) — התאמה: {row['score']*100:.1f}%"
        with st.expander(header):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**ניסיון:** {row.get('experience', '')}")
                st.write(f"**תחומי עניין:** {row.get('interests', '')}")
                st.write(f"**מחוז / מגזר:** {row.get('region', '')} / {row.get('sector', '')}")
            with c2:
                if mode == "Semantic Search":
                    st.write("**למה נמצאה ההתאמה:**")
                    for label in explain_match(query, row):
                        st.write(f"✔ {label}")
                if role == "צוות מעוז (Admin)" and row.get("internal_note"):
                    st.write(f"🔒 **הערה פנימית (Admin בלבד):** {row['internal_note']}")

            if st.button("🔗 מצא/י אנשים דומים", key=f"similar-{idx}"):
                similar = find_similar(idx, embeddings, df, top_k=5)
                st.write("**פרופילים דומים:**")
                for _, srow in similar.iterrows():
                    st.write(f"- {srow['name']} ({srow['title']}, {srow['organization']}) — {srow['score']*100:.1f}%")
else:
    st.info("הזן שאילתה כדי להתחיל בחיפוש.")
