# -*- coding: utf-8 -*-
"""
engine.py — Search Engine Core
מנוע החיפוש הסמנטי: טעינת נתונים, Embeddings, חיפוש ודירוג.

מודול זה אינו תלוי ב-Streamlit ואפשר לייבא אותו מכל ממשק.
"""

import re
import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# מיפוי שמות עמודות אפשריים בקובץ הגלם -> שם קנוני בקוד.
# הוסיפו כאן כל שם עמודה נוסף שקיים בקובץ ה-CSV שלכם.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "name": ["name", "full_name", "שם", "שם מלא"],
    "title": ["title", "role", "position", "תפקיד"],
    "organization": ["organization", "org", "company", "ארגון"],
    "region": ["region", "district", "מחוז", "אזור"],
    "sector": ["sector", "מגזר"],
    "experience": ["experience", "bio", "about", "ניסיון", "תקציר", "רקע מקצועי"],
    "interests": ["interests", "tags", "תחומי עניין"],
}


# --------------------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------------------

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
    """בונה טקסט מייצג לכל פרופיל, לצורך חישוב Embedding."""
    return (
        f"שם: {row.get('name', '')}. "
        f"תפקיד: {row.get('title', '')}. "
        f"ארגון: {row.get('organization', '')}. "
        f"מחוז: {row.get('region', '')}. "
        f"מגזר: {row.get('sector', '')}. "
        f"ניסיון: {row.get('experience', '')}. "
        f"תחומי עניין: {row.get('interests', '')}."
    )


def load_data(csv_path: str) -> pd.DataFrame:
    """טוען CSV, מנרמל עמודות, מוסיף שדות RBAC ובונה טקסט חיפוש."""
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    df = add_demo_rbac_fields(df)
    df["search_text"] = df.apply(build_search_text, axis=1)
    return df


def file_hash_of(path: str) -> str:
    """מחשב MD5 hash של הקובץ לצורכי cache-busting."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "missing"


# --------------------------------------------------------------------------------------
# Model & Embeddings
# --------------------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    """טוען את מודל ה-Embedding."""
    return SentenceTransformer(MODEL_NAME)


def embed_corpus(model: SentenceTransformer, texts: list) -> np.ndarray:
    """מחשב Embeddings לכל הפרופילים."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """מחשב Embedding לשאילתה בודדת."""
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------

def semantic_search(query_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """חיפוש סמנטי — Cosine Similarity בין וקטור השאילתה לכל הפרופילים."""
    return np.dot(embeddings, query_vec)


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
    """מוצא פרופילים דומים לפרופיל נתון לפי קרבה ב-Vector Space."""
    target = embeddings[idx]
    scores = np.dot(embeddings, target)
    order = np.argsort(-scores)
    order = [i for i in order if i != idx][:top_k]
    result = df.iloc[order].copy()
    result["score"] = scores[order]
    return result


# --------------------------------------------------------------------------------------
# RBAC & Filtering
# --------------------------------------------------------------------------------------

def apply_rbac_filter(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """סינון לפי הרשאות — מתבצע לפני הצגת תוצאות, לא אחרי."""
    if role == "חבר/ת רשת":
        return df[df["is_sensitive"] != True].copy()  # noqa: E712
    return df.copy()


def apply_metadata_filters(
    df: pd.DataFrame,
    region_filter: list,
    sector_filter: list,
) -> pd.DataFrame:
    """סינון לפי מחוז ומגזר."""
    if region_filter:
        df = df[df["region"].isin(region_filter)]
    if sector_filter:
        df = df[df["sector"].isin(sector_filter)]
    return df


def score_level(score: float) -> str:
    """מחזיר רמת ציון לצורך עיצוב (CSS class)."""
    if score >= 0.55:
        return "high"
    elif score >= 0.35:
        return "medium"
    return "low"
