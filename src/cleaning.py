"""
Carga companies.xlsx, normaliza y limpia el dataset al esquema canónico.
Script de producción: sin red, sin embeddings, solo preparación de datos.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "companies.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUT_CLEAN = OUTPUT_DIR / "companies_clean.csv"
OUT_MANUAL = OUTPUT_DIR / "companies_manual_review_duplicates.csv"
OUT_REPORT = OUTPUT_DIR / "cleaning_report.json"

# Columnas fuente posibles (variantes de acentos/espacios se resuelven por normalización)
SOURCE_COLUMN_SPECS: dict[str, tuple[str, ...]] = {
    "id": ("ID", "Id"),
    "fecha": ("Fecha",),
    "empresa": ("Empresa",),
    "descripcion": ("Descripción", "Descripcion"),
    "tamano": ("Tamaño", "Tamano"),
    "fondeo": ("Fondeo",),
    "ingresos": ("Ingresos",),
    "valuacion": ("Valuación", "Valuacion"),
    "oportunidades": ("Oportunidades",),
    "sede": ("Sede",),
    "alianzas": ("Alianzas",),
    "mercados": ("Mercados",),
    "pagina_web": ("Página Web", "Pagina Web"),
    "facebook": ("Facebook",),
    "twitter": ("Twitter",),
    "instagram": ("Instagram",),
    "linkedin": ("LinkedIn", "Linkedin"),
    "youtube": ("YouTube", "Youtube"),
    "video": ("Video",),
}

NULL_STRINGS_LOWER = frozenset(
    {
        "n/d",
        "nd",
        "n.a.",
        "n/a",
        "na",
        "null",
        "none",
        "no aplica",
        "no disponible",
        "información no disponible",
        "informacion no disponible",
        "s/d",
    }
)

CANONICAL_COLUMNS: list[str] = [
    "company_id",
    "created_date_raw",
    "created_date_clean",
    "company_name_raw",
    "company_name_display",
    "company_name_normalized",
    "description_raw",
    "description_clean",
    "size_raw",
    "size_clean",
    "funding_raw",
    "revenue_raw",
    "valuation_raw",
    "opportunities_raw",
    "opportunities_clean",
    "location_raw",
    "location_clean",
    "alliances_raw",
    "alliances_clean",
    "sector_raw",
    "sector_list_clean",
    "sector_primary",
    "website_raw",
    "website_clean",
    "website_domain",
    "facebook_clean",
    "twitter_clean",
    "instagram_clean",
    "linkedin_clean",
    "youtube_clean",
    "video_clean",
    "social_links_clean",
    "semantic_text",
    "missing_description_flag",
    "invalid_website_flag",
    "multi_sector_flag",
    "duplicate_suspect_flag",
    "low_information_flag",
    "text_noise_flag",
    "url_noise_flag",
    "record_status",
    "duplicate_action",
]

TRACKING_QUERY_KEYS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "igshid",
        "si",
    }
)

SIZE_CANONICAL_ORDER: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^\s*0\s*[-–—a]\s*10\b",
            re.I,
        ),
        "0 a 10 Empleados",
    ),
    (
        re.compile(r"^\s*11\s*[-–—a]\s*50\b", re.I),
        "11 a 50 Empleados",
    ),
    (
        re.compile(r"^\s*51\s*[-–—a]\s*200\b", re.I),
        "51 a 200 Empleados",
    ),
    (
        re.compile(r"^\s*201\s*[-–—a]\s*500\b", re.I),
        "201 a 500 Empleados",
    ),
    (
        re.compile(r"^\s*501\s*[-–—a]\s*1[\.,]?000\b", re.I),
        "501 a 1,000 Empleados",
    ),
    (
        re.compile(r"^\s*1[\.,]?001\s*[-–—a]\s*5[\.,]?000\b", re.I),
        "1,001 a 5,000 Empleados",
    ),
    (
        re.compile(r"^\s*5[\.,]?001\s*[-–—a]\s*10[\.,]?000\b", re.I),
        "5,001 a 10,000 Empleados",
    ),
    (
        re.compile(
            r"^\s*(\+\s*|más\s+|mas\s+)?(de\s+)?10[\.,]?000|10000\+|\+?\s*10000",
            re.I,
        ),
        "+ de 10,000 Empleados",
    ),
]


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _normalize_header_key(s: str) -> str:
    s = " ".join(str(s).strip().split())
    return _strip_accents(s).lower()


def load_companies_file(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    logger.info("Archivo cargado: %s — filas=%s columnas=%s", path.name, len(df), len(df.columns))
    return df


def map_columns(columns: pd.Index) -> dict[str, str | None]:
    col_list = [str(c) for c in columns]
    exact = {c: c for c in col_list}
    normalized_to_actual: dict[str, str] = {}
    for c in col_list:
        nk = _normalize_header_key(c)
        if nk not in normalized_to_actual:
            normalized_to_actual[nk] = c

    resolved: dict[str, str | None] = {}
    for key, candidates in SOURCE_COLUMN_SPECS.items():
        found: str | None = None
        for cand in candidates:
            if cand in exact:
                found = cand
                break
        if found is None:
            for cand in candidates:
                nk = _normalize_header_key(cand)
                if nk in normalized_to_actual:
                    found = normalized_to_actual[nk]
                    break
        resolved[key] = found
    return resolved


def _is_null_like_string(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    if all(c in "-–—." for c in t):
        return True
    if all(c.isspace() for c in t):
        return True
    if all(c == "/" for c in t):
        return True
    if t.lower() in NULL_STRINGS_LOWER:
        return True
    return False


def normalize_nulls_scalar(val: Any, *, treat_zero_as_null: bool = False) -> Any:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        if treat_zero_as_null and int(val) == 0:
            return np.nan
        return val
    if isinstance(val, (float, np.floating)) and not isinstance(val, bool):
        if np.isnan(val):
            return np.nan
        if treat_zero_as_null and float(val) == 0.0:
            return np.nan
        return val
    s = str(val).strip()
    if _is_null_like_string(s):
        return np.nan
    if treat_zero_as_null and s == "0":
        return np.nan
    return val


def normalize_nulls_series(series: pd.Series, *, treat_zero_as_null: bool = False) -> pd.Series:
    return series.map(lambda x: normalize_nulls_scalar(x, treat_zero_as_null=treat_zero_as_null))


def _remove_markdown_noise(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def clean_text_field(text: Any, *, aggressive_name: bool = False) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")
    s = _remove_markdown_noise(s)
    s = re.sub(r"[-—–]{4,}", " ", s)
    s = re.sub(r"_{4,}", " ", s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip()
    if not aggressive_name:
        s = re.sub(r"\s+", " ", s)
    return s


def normalize_company_name_display(raw: Any) -> str:
    s = clean_text_field(raw)
    return s


def normalize_company_name_normalized(display: str) -> str:
    if not display:
        return ""
    s = display.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_urls(raw: str) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    s = str(raw).strip()
    parts: list[str] = []
    if re.search(r"https?://", s, re.I):
        for m in re.finditer(r"https?://[^\s,;|]+", s, re.I):
            parts.append(m.group(0).rstrip(").,;]}\"'"))
    if parts:
        return parts
    for sep in ["|", ";", "\n"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    if not parts:
        parts = [s]
    return parts


def _strip_tracking_params(parsed) -> tuple:
    if not parsed.query:
        return parsed
    qs = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {
        k: v
        for k, v in qs.items()
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_QUERY_KEYS
    }
    new_query = urlencode(filtered, doseq=True) if filtered else ""
    return parsed._replace(query=new_query)


def _is_valid_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if not p.netloc or "." not in p.netloc:
            return False
        return True
    except Exception:
        return False


def clean_url_field(
    raw: Any,
    *,
    is_website: bool = False,
) -> tuple[str | float, str | float, str | float, int]:
    """Retorna (clean, domain_or_nan, raw_str, url_noise_flag 0/1)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return (np.nan, np.nan, np.nan, 0)
    s0 = str(raw).strip()
    raw_display = s0 if s0 else np.nan
    if _is_null_like_string(s0):
        return (np.nan, np.nan, raw_display if pd.notna(raw_display) else np.nan, 0)

    noise = 0
    if len(re.findall(r"https?://", s0, re.I)) > 1:
        noise = 1
    if re.search(r"utm_|fbclid|gclid", s0, re.I):
        noise = 1
    if re.search(r"[^\s]+\s+https?://", s0):
        noise = 1

    candidates = _split_urls(s0)
    chosen = ""
    for c in candidates:
        c = c.strip()
        if not c.startswith(("http://", "https://")):
            c = "https://" + c.lstrip("/")
        try:
            p = urlparse(c)
            p2 = _strip_tracking_params(p)
            clean = urlunparse(
                (
                    p2.scheme or "https",
                    p2.netloc.lower().rstrip("."),
                    p2.path.rstrip("/") or "/",
                    "",
                    p2.query,
                    "",
                )
            )
            if clean.endswith("/") and p2.path in ("", "/"):
                clean = clean.rstrip("/")
            if _is_valid_http_url(clean):
                chosen = clean
                break
        except Exception:
            continue

    if not chosen:
        return (np.nan, np.nan, raw_display, max(noise, 1))

    dom = urlparse(chosen).netloc.lower()
    if dom.startswith("www."):
        dom = dom[4:]
    return (chosen, dom if dom else np.nan, raw_display, noise)


def parse_created_date(raw: Any) -> tuple[Any, pd.Timestamp | float, bool]:
    """(raw_repr, ts o NaT, parse_ok)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return (np.nan, pd.NaT, True)
    if isinstance(raw, pd.Timestamp):
        return (raw, raw.normalize(), True)
    if isinstance(raw, datetime.datetime):
        ts = pd.Timestamp(raw)
        return (raw, ts.normalize(), True)
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        dt = pd.to_datetime(raw, errors="coerce")
        ok = not pd.isna(dt)
        return (raw, dt.normalize() if ok else pd.NaT, ok)
    s = str(raw).strip()
    if _is_null_like_string(s):
        return (np.nan, pd.NaT, True)
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce")
    ok = not pd.isna(dt)
    return (s if s else raw, dt.normalize() if ok else pd.NaT, ok)


def normalize_sector_field(raw: Any) -> tuple[str, str, str, int]:
    """sector_raw, sector_list_clean, sector_primary, multi_sector_flag."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ("", "", "", 0)
    s = clean_text_field(raw)
    if not s or _is_null_like_string(s):
        return ("", "", "", 0)
    sector_raw = str(raw).strip() if raw is not None else ""
    parts = [clean_text_field(p) for p in re.split(r",|;", str(raw))]
    parts = [p for p in parts if p and not _is_null_like_string(p)]
    cleaned: list[str] = []
    seen: set[str] = set()
    for p in parts:
        t = " ".join(w.capitalize() if w else "" for w in p.split())
        t = re.sub(r"\s+", " ", t).strip()
        key = t.lower()
        if key and key not in seen:
            seen.add(key)
            cleaned.append(t)
    if not cleaned:
        t = " ".join(w.capitalize() for w in s.split())
        cleaned = [t] if t else []
    primary = cleaned[0] if cleaned else ""
    multi = 1 if len(cleaned) > 1 else 0
    list_str = " | ".join(cleaned) if cleaned else ""
    return (sector_raw, list_str, primary, multi)


def normalize_location_field(raw: Any) -> tuple[str, str]:
    loc_raw = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
    c = clean_text_field(raw)
    return (loc_raw, c)


def normalize_size_field(raw: Any) -> tuple[str, str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ("", "")
    sr = str(raw).strip()
    s = clean_text_field(raw)
    if not s:
        return (sr, "")
    s_cmp = re.sub(r"\s+", " ", s)
    for pat, label in SIZE_CANONICAL_ORDER:
        if pat.search(s_cmp):
            return (sr, label)
    return (sr, s_cmp)


def _financial_raw(val: Any) -> Any:
    v = normalize_nulls_scalar(val, treat_zero_as_null=False)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return v


def build_semantic_text(
    name_disp: str,
    desc_clean: str,
    sector_primary: str,
    sector_list: str,
    opp_clean: str,
    all_clean: str,
) -> str:
    def is_placeholder(x: str) -> bool:
        if not x or not str(x).strip():
            return True
        t = str(x).strip().lower()
        return t in NULL_STRINGS_LOWER or _is_null_like_string(t)

    parts: list[str] = []
    if not is_placeholder(name_disp):
        parts.append(name_disp.strip())
    if not is_placeholder(desc_clean):
        parts.append(desc_clean.strip())
    sec_line = ""
    if not is_placeholder(sector_primary):
        sec_line = sector_primary.strip()
    elif not is_placeholder(sector_list):
        sec_line = sector_list.strip()
    if sec_line:
        parts.append(sec_line)
    if not is_placeholder(opp_clean):
        parts.append(opp_clean.strip())
    if not is_placeholder(all_clean):
        parts.append(all_clean.strip())
    # dedupe consecutive identical chunks
    out: list[str] = []
    prev = None
    for p in parts:
        key = p.lower()
        if key != prev:
            out.append(p)
            prev = key
    return "\n\n".join(out)


def _text_noise_heuristic(raw: str, clean: str) -> int:
    if not raw:
        return 0
    r, c = str(raw), clean or ""
    score = 0
    if len(re.findall(r"\n{3,}", r)) > 0:
        score += 1
    if re.search(r"[`*#]{3,}", r):
        score += 1
    if re.search(r"_{4,}|[-—–]{5,}", r):
        score += 1
    if len(c) > 0 and len(r) > len(c) * 1.4:
        score += 1
    return 1 if score >= 2 else 0


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_probable_duplicates(
    df: pd.DataFrame,
    name_col: str = "company_name_normalized",
    loc_col: str = "location_clean",
    sec_col: str = "sector_primary",
    desc_col: str = "description_clean",
) -> set[int]:
    """Índices (posición 0..n-1) marcados como sospechosos (nivel 4)."""
    idxs = df.index.tolist()
    suspect: set[int] = set()

    blocks: dict[tuple[str, str], list[int]] = {}
    for i, ix in enumerate(idxs):
        row = df.loc[ix]
        loc = str(row.get(loc_col, "") or "").strip().lower()
        sec = str(row.get(sec_col, "") or "").strip().lower()
        if not loc or not sec:
            continue
        key = (sec, loc[: min(80, len(loc))])
        blocks.setdefault(key, []).append(i)

    for _k, positions in blocks.items():
        if len(positions) < 2:
            continue
        m = len(positions)
        for a in range(m):
            for b in range(a + 1, m):
                i, j = positions[a], positions[b]
                ix_i, ix_j = idxs[i], idxs[j]
                ri, rj = df.loc[ix_i], df.loc[ix_j]
                ni = str(ri.get(name_col, "") or "")
                nj = str(rj.get(name_col, "") or "")
                if not ni or not nj:
                    continue
                if ni == nj:
                    continue
                if _sim(ni, nj) < 0.88:
                    continue
                di = str(ri.get(desc_col, "") or "")
                dj = str(rj.get(desc_col, "") or "")
                if not di or not dj:
                    continue
                if _sim(di, dj) < 0.90:
                    continue
                suspect.add(i)
                suspect.add(j)

    return suspect


def dedupe_exact_levels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Niveles 1–3. L2 y L3 no eliminan filas con duplicate_suspect_flag (revisión manual).
    """
    stats = {
        "by_id": 0,
        "by_name": 0,
        "by_domain": 0,
    }
    work = df.copy()
    work["_orig_order"] = np.arange(len(work))

    def completeness_score(row: pd.Series) -> int:
        cols = [
            "description_clean",
            "website_clean",
            "sector_primary",
            "opportunities_clean",
            "location_clean",
        ]
        s = 0
        for c in cols:
            v = row.get(c)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                if str(v).strip():
                    s += 1
        return s

    def pick_canonical(sub: pd.DataFrame) -> Any:
        scores = sub.apply(completeness_score, axis=1)
        mx = scores.max()
        candidates = sub.index[scores == mx]
        return sub.loc[candidates, "_orig_order"].idxmin()

    # Nivel 1: company_id
    cid = work["company_id"]
    mask_id = cid.notna() & (cid.astype(str).str.len() > 0)
    drop_idx: list = []
    for val in work.loc[mask_id, "company_id"].unique():
        sub = work[work["company_id"] == val]
        if len(sub) <= 1:
            continue
        keep = pick_canonical(sub)
        for ix in sub.index:
            if ix != keep:
                drop_idx.append(ix)
        stats["by_id"] += len(sub) - 1
    work = work.drop(index=drop_idx, errors="ignore")

    # Nivel 2: company_name_normalized (no quitar sospechosos de duplicado probable)
    drop_idx = []
    for val in work["company_name_normalized"].dropna().unique():
        if not str(val).strip():
            continue
        sub = work[work["company_name_normalized"] == val]
        if len(sub) <= 1:
            continue
        if (sub["duplicate_suspect_flag"] == 1).any():
            continue
        keep = pick_canonical(sub)
        for ix in sub.index:
            if ix != keep:
                drop_idx.append(ix)
        stats["by_name"] += len(sub) - 1
    work = work.drop(index=drop_idx, errors="ignore")

    # Nivel 3: website_domain
    drop_idx = []
    dom_series = work["website_domain"]
    for val in dom_series.dropna().unique():
        ds = str(val).strip().lower()
        if not ds:
            continue
        sub = work[work["website_domain"].astype(str).str.lower() == ds]
        if len(sub) <= 1:
            continue
        if (sub["duplicate_suspect_flag"] == 1).any():
            continue
        keep = pick_canonical(sub)
        for ix in sub.index:
            if ix != keep:
                drop_idx.append(ix)
        stats["by_domain"] += len(sub) - 1
    work = work.drop(index=drop_idx, errors="ignore")

    work = work.drop(columns=["_orig_order"], errors="ignore")
    return work.reset_index(drop=True), stats


def validate_quality_row(row: pd.Series) -> tuple[str, int, int, int]:
    """record_status, missing_description_flag, invalid_website_flag, low_information_flag."""
    nd = row.get("company_name_display")
    if nd is None or (isinstance(nd, float) and pd.isna(nd)):
        name = ""
    else:
        name = str(nd).strip()
    dd = row.get("description_clean")
    if dd is None or (isinstance(dd, float) and pd.isna(dd)):
        desc = ""
    else:
        desc = str(dd).strip()
    web = row.get("website_clean")
    web_ok = web is not None and not (isinstance(web, float) and pd.isna(web)) and str(web).strip() != ""
    sp = row.get("sector_primary")
    if sp is None or (isinstance(sp, float) and pd.isna(sp)):
        sec = ""
    else:
        sec = str(sp).strip()
    w_raw = row.get("website_raw")

    missing_desc = 1 if len(desc) < 12 else 0
    invalid_web = 0
    if w_raw is not None and not (isinstance(w_raw, float) and pd.isna(w_raw)) and str(w_raw).strip():
        if not web_ok:
            invalid_web = 1
    elif int(row.get("invalid_website_flag", 0) or 0) == 1:
        invalid_web = 1

    has_core = bool(name) and (len(desc) >= 12 or web_ok or bool(sec))

    if not name:
        status = "invalid_record"
    elif has_core:
        status = "valid_record"
    else:
        status = "partial_record"

    raw_d = row.get("created_date_raw")
    dclean = row.get("created_date_clean")
    bad_date = dclean is None or (isinstance(dclean, float) and pd.isna(dclean)) or (
        isinstance(dclean, pd.Timestamp) and pd.isna(dclean)
    )
    if bad_date and raw_d is not None and not (isinstance(raw_d, float) and pd.isna(raw_d)):
        if str(raw_d).strip():
            if status == "valid_record":
                status = "partial_record"

    low_info = 0
    if name and status != "invalid_record":
        info_score = (1 if len(desc) >= 12 else 0) + (1 if web_ok else 0) + (1 if len(sec) >= 2 else 0)
        if info_score <= 1:
            low_info = 1
        if status == "partial_record" and info_score == 0:
            low_info = 1

    return status, missing_desc, invalid_web, low_info


def apply_validation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vr = out.apply(validate_quality_row, axis=1, result_type="expand")
    vr.columns = ["record_status", "missing_description_flag", "invalid_website_flag", "low_information_flag"]
    for c in vr.columns:
        out[c] = vr[c].values
    return out


def export_files(
    df_final: pd.DataFrame,
    df_manual: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_CLEAN, index=False, encoding="utf-8")
    df_manual.to_csv(OUT_MANUAL, index=False, encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_report(
    n_in: int,
    df_out: pd.DataFrame,
    df_manual: pd.DataFrame,
    dedup_stats: dict[str, int],
    n_exact_total: int,
) -> dict[str, Any]:
    rs = df_out["record_status"].value_counts()
    return {
        "total_rows_input": n_in,
        "total_rows_output": int(len(df_out)),
        "total_invalid_records": int(rs.get("invalid_record", 0)),
        "total_partial_records": int(rs.get("partial_record", 0)),
        "total_valid_records": int(rs.get("valid_record", 0)),
        "total_exact_duplicates_removed": int(n_exact_total),
        "total_probable_duplicates_flagged": int(df_out["duplicate_suspect_flag"].sum()),
        "total_invalid_websites": int(df_out["invalid_website_flag"].sum()),
        "total_missing_descriptions": int(df_out["missing_description_flag"].sum()),
        "total_multi_sector_records": int(df_out["multi_sector_flag"].sum()),
        "total_low_information_records": int(df_out["low_information_flag"].sum()),
        "total_rows_sent_to_manual_review": int(len(df_manual)),
        "dedup_by_id": dedup_stats.get("by_id", 0),
        "dedup_by_name": dedup_stats.get("by_name", 0),
        "dedup_by_domain": dedup_stats.get("by_domain", 0),
    }


def _series(df: pd.DataFrame, col_map: dict[str, str | None], key: str) -> pd.Series:
    col = col_map.get(key)
    if not col:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[col]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df_in = load_companies_file(DATA_PATH)
    n_input = len(df_in)
    col_map = map_columns(df_in.columns)

    # Series crudas con nulos normalizados (texto: cero como null solo donde aplique)
    def col_series(key: str, zero_placeholder: bool = False) -> pd.Series:
        s = _series(df_in, col_map, key)
        return normalize_nulls_series(s, treat_zero_as_null=zero_placeholder)

    emp_raw = col_series("empresa", zero_placeholder=True)
    desc_raw = col_series("descripcion", zero_placeholder=True)
    tam_raw = col_series("tamano", zero_placeholder=True)
    fund_raw = col_series("fondeo").map(_financial_raw)
    ing_raw = col_series("ingresos").map(_financial_raw)
    val_raw = col_series("valuacion").map(_financial_raw)
    opor_raw = col_series("oportunidades", zero_placeholder=True)
    sede_raw = col_series("sede", zero_placeholder=True)
    alian_raw = col_series("alianzas", zero_placeholder=True)
    merc_raw = col_series("mercados", zero_placeholder=True)
    web_raw = col_series("pagina_web", zero_placeholder=True)
    fb_raw = col_series("facebook", zero_placeholder=True)
    tw_raw = col_series("twitter", zero_placeholder=True)
    ig_raw = col_series("instagram", zero_placeholder=True)
    li_raw = col_series("linkedin", zero_placeholder=True)
    yt_raw = col_series("youtube", zero_placeholder=True)
    vid_raw = col_series("video", zero_placeholder=True)
    id_raw = col_series("id")
    fecha_raw = col_series("fecha")

    rows: list[dict[str, Any]] = []
    for i in range(n_input):
        er = emp_raw.iloc[i]
        empresa_str = "" if pd.isna(er) else str(er).strip()
        name_disp = normalize_company_name_display(er)
        name_norm = normalize_company_name_normalized(name_disp)

        dr = desc_raw.iloc[i]
        description_raw = np.nan if pd.isna(dr) else dr
        desc_clean = clean_text_field(dr)

        sr = tam_raw.iloc[i]
        size_raw, size_clean = normalize_size_field(sr)

        fr = fund_raw.iloc[i]
        funding_raw = fr

        ingr = ing_raw.iloc[i]
        revenue_raw = ingr

        vr = val_raw.iloc[i]
        valuation_raw = vr

        opr = opor_raw.iloc[i]
        opportunities_raw = np.nan if pd.isna(opr) else opr
        opportunities_clean = clean_text_field(opr)

        sed = sede_raw.iloc[i]
        location_raw, location_clean = normalize_location_field(sed)

        alr = alian_raw.iloc[i]
        alliances_raw = np.nan if pd.isna(alr) else alr
        alliances_clean = clean_text_field(alr)

        mer = merc_raw.iloc[i]
        sector_raw, sector_list_clean, sector_primary, multi_sector_flag = normalize_sector_field(mer)

        wr = web_raw.iloc[i]
        w_clean, w_dom, w_raw_out, w_noise = clean_url_field(wr, is_website=True)

        fb_c, _, _, fb_n = clean_url_field(fb_raw.iloc[i])
        tw_c, _, _, tw_n = clean_url_field(tw_raw.iloc[i])
        ig_c, _, _, ig_n = clean_url_field(ig_raw.iloc[i])
        li_c, _, _, li_n = clean_url_field(li_raw.iloc[i])
        yt_c, _, _, yt_n = clean_url_field(yt_raw.iloc[i])
        vid_c, _, _, vid_n = clean_url_field(vid_raw.iloc[i])

        url_noise = max(w_noise, fb_n, tw_n, ig_n, li_n, yt_n, vid_n)

        social_urls: list[str] = []
        for u in (fb_c, tw_c, ig_c, li_c, yt_c):
            if u is not None and not (isinstance(u, float) and pd.isna(u)):
                social_urls.append(str(u))
        social_links_clean = json.dumps(social_urls, ensure_ascii=False)

        id_val = id_raw.iloc[i]
        if pd.isna(id_val):
            company_id = np.nan
        elif isinstance(id_val, float) and id_val == int(id_val):
            company_id = str(int(id_val))
        else:
            company_id = str(id_val).strip()
            if not company_id:
                company_id = np.nan

        raw_d, dt_clean, d_ok = parse_created_date(fecha_raw.iloc[i])
        created_date_raw = raw_d
        created_date_clean = dt_clean if d_ok else pd.NaT

        text_noise = _text_noise_heuristic(
            str(description_raw) if pd.notna(description_raw) else "",
            desc_clean,
        )

        invalid_website_flag = 1 if (wr is not None and not pd.isna(wr) and str(wr).strip() and (pd.isna(w_clean) or w_clean is None)) else 0

        sem = build_semantic_text(
            name_disp,
            desc_clean,
            sector_primary,
            sector_list_clean,
            opportunities_clean,
            alliances_clean,
        )

        rows.append(
            {
                "company_id": company_id,
                "created_date_raw": created_date_raw,
                "created_date_clean": created_date_clean,
                "company_name_raw": empresa_str if empresa_str else np.nan,
                "company_name_display": name_disp if name_disp else np.nan,
                "company_name_normalized": name_norm if name_norm else np.nan,
                "description_raw": description_raw,
                "description_clean": desc_clean if desc_clean else np.nan,
                "size_raw": size_raw if size_raw else np.nan,
                "size_clean": size_clean if size_clean else np.nan,
                "funding_raw": funding_raw,
                "revenue_raw": revenue_raw,
                "valuation_raw": valuation_raw,
                "opportunities_raw": opportunities_raw,
                "opportunities_clean": opportunities_clean if opportunities_clean else np.nan,
                "location_raw": location_raw if location_raw else np.nan,
                "location_clean": location_clean if location_clean else np.nan,
                "alliances_raw": alliances_raw,
                "alliances_clean": alliances_clean if alliances_clean else np.nan,
                "sector_raw": sector_raw if sector_raw else np.nan,
                "sector_list_clean": sector_list_clean if sector_list_clean else np.nan,
                "sector_primary": sector_primary if sector_primary else np.nan,
                "website_raw": w_raw_out,
                "website_clean": w_clean,
                "website_domain": w_dom,
                "facebook_clean": fb_c,
                "twitter_clean": tw_c,
                "instagram_clean": ig_c,
                "linkedin_clean": li_c,
                "youtube_clean": yt_c,
                "video_clean": vid_c,
                "social_links_clean": social_links_clean,
                "semantic_text": sem if sem.strip() else np.nan,
                "missing_description_flag": 0,
                "invalid_website_flag": invalid_website_flag,
                "multi_sector_flag": multi_sector_flag,
                "duplicate_suspect_flag": 0,
                "low_information_flag": 0,
                "text_noise_flag": text_noise,
                "url_noise_flag": url_noise,
                "record_status": "valid_record",
                "duplicate_action": "kept",
            }
        )

    enriched = pd.DataFrame(rows)
    enriched = apply_validation_dataframe(enriched)

    # Nivel 4: duplicados probables (antes de eliminar exactos)
    positions = detect_probable_duplicates(enriched)
    for pos in positions:
        enriched.iloc[pos, enriched.columns.get_loc("duplicate_suspect_flag")] = 1
        enriched.iloc[pos, enriched.columns.get_loc("duplicate_action")] = "manual_review"

    deduped, dstats = dedupe_exact_levels(enriched)
    n_exact = dstats["by_id"] + dstats["by_name"] + dstats["by_domain"]

    # Registros conservados con acción kept si no eran manual_review
    for i in range(len(deduped)):
        if deduped.iloc[i]["duplicate_action"] != "manual_review":
            deduped.iloc[i, deduped.columns.get_loc("duplicate_action")] = "kept"

    deduped = apply_validation_dataframe(deduped)

    manual = deduped[
        (deduped["duplicate_suspect_flag"] == 1) | (deduped["duplicate_action"] == "manual_review")
    ].copy()

    for c in CANONICAL_COLUMNS:
        if c not in deduped.columns:
            deduped[c] = np.nan
    deduped = deduped[CANONICAL_COLUMNS]

    for c in CANONICAL_COLUMNS:
        if c not in manual.columns:
            manual[c] = np.nan
    manual = manual[CANONICAL_COLUMNS]

    report = generate_report(n_input, deduped, manual, dstats, n_exact)

    export_files(deduped, manual, report)

    print(
        f"""
Resumen de limpieza
-------------------
Filas leídas:              {report['total_rows_input']}
Filas de salida:           {report['total_rows_output']}
  · Válidas:               {report['total_valid_records']}
  · Parciales:             {report['total_partial_records']}
  · Inválidas:             {report['total_invalid_records']}
Duplicados exactos elim.:  {report['total_exact_duplicates_removed']}
  (por ID / nombre / dominio: {report['dedup_by_id']} / {report['dedup_by_name']} / {report['dedup_by_domain']})
Duplicados prob. (revisión): {report['total_probable_duplicates_flagged']}
Websites inválidos:        {report['total_invalid_websites']}
Archivo principal:         {OUT_CLEAN.resolve()}
Revisión manual:           {OUT_MANUAL.resolve()}
Reporte JSON:              {OUT_REPORT.resolve()}
"""
    )
    logger.info("Proceso terminado.")


if __name__ == "__main__":
    main()
