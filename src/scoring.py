"""
Radar scoring: thematic clustering from precomputed embeddings, strategic sub-scores, exports.
Does not regenerate embeddings, query the web, or run semantic search.
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# --- Paths (project root = parent of src/) ---
# --- Paths (project root = parent of src/) ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

CLEAN_CSV = OUTPUT_DIR / "cleaned_companies.csv"
EMB_NPY = OUTPUT_DIR / "company_embeddings.npy"
OUT_CSV = OUTPUT_DIR / "scored_companies.csv"
REPORT_JSON = OUTPUT_DIR / "scoring_report.json"
CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "cluster_summary.csv"

REQUIRED_CLEAN_COLS = (
    "company_id",
    "company_name_display",
    "company_name_normalized",
    "description_clean",
    "sector_primary",
    "sector_list_clean",
    "website_clean",
    "website_domain",
    "semantic_text",
    "record_status",
    "missing_description_flag",
    "invalid_website_flag",
    "low_information_flag",
)



VALID_RECORD_STATUS = frozenset({"valid_record", "partial_record"})

# Priority industries (display labels) -> folded keyword tuples
PRIORITY_INDUSTRY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Telecomunicaciones", ("telecom", "telecomunicacion", "telefonia", "carrier", "5g", "4g", "mvno", "isp")),
    ("Banca", ("banca", "banking", "banco ", " banco", "creditos bancarios", "core banking")),
    ("Fintech", ("fintech", "pagos digitales", "payment", "wallet", "neobank", "open banking")),
    ("Salud", ("salud", "health", "hospital", "clinica", "telemedicina", "paciente", "medico")),
    ("Educación", ("educacion", "edtech", "universidad", "formacion", "e-learning", "learning")),
    ("Medios y Entretenimiento", ("medios", "entretenimiento", "streaming", "broadcast", "contenido digital")),
    ("Música", ("musica", "music", "audio", "sonido", "sello discografico")),
    ("Agricultura", ("agricultura", "agtech", "agro", "cultivo", "ganaderia")),
    ("Minería", ("mineria", "mining", "mina", "metalurgia", "extraccion minera")),
    ("Energía", ("energia", "energetico", "oil", "gas", "renovable", "solar", "eolica")),
)

# Thematic labels for clusters / theme_tags (folded keywords)
THEME_TAG_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Healthtech", ("salud", "health", "hospital", "clinica", "telemedicina", "paciente")),
    ("Fintech", ("fintech", "neobank", "pagos", "payment", "wallet", "open banking")),
    ("Edtech", ("edtech", "educacion", "e-learning", "universidad", "formacion")),
    ("CX Analytics", ("customer experience", "cx ", " cx", "journey", "analitica cx")),
    ("Contact Center AI", ("contact center", "call center", "centro de contacto", "ivr")),
    ("Automatización hospitalaria", ("hospital", "clinica", "admision hospital", "historia clinica")),
    ("RPA / Process Automation", ("rpa", "automatizacion", "workflow", "orquestacion", "bpm")),
    ("Identity & Access", ("identity", "iam", "sso", "autenticacion", "autorizacion", "mfa")),
    ("Telecom Enablement", ("telecom", "carrier", "operador", "bss", "oss", "mvno")),
    ("E-commerce Enablement", ("ecommerce", "e-commerce", "marketplace", "tienda online", "retail digital")),
    ("IoT / Edge", ("iot", "edge computing", "sensor", "dispositivo conectado")),
    ("Data Infrastructure", ("data lake", "warehouse", "etl", "pipeline de datos", "big data")),
    ("Customer Engagement", ("engagement", "fidelizacion", "crm", "retencion de clientes")),
    ("Security / Zero Trust", ("ciberseguridad", "zero trust", "seguridad informatica", "threat")),
    ("General Technology", ("software", "plataforma", "tecnologia", "solucion", "saas")),
)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def load_clean_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"ERROR: No se encontró el dataset limpio: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        sys.exit(
            f"ERROR: No existe el archivo de embeddings requerido: {path}. "
            "Ejecute primero el pipeline de embeddings."
        )
    try:
        arr = np.load(path)
    except Exception as e:
        sys.exit(f"ERROR: No se pudieron cargar los embeddings desde {path}: {e}")

    if arr.ndim != 2:
        sys.exit(
            f"ERROR: Se esperaban embeddings 2D con forma (n, d), "
            f"pero se obtuvo shape={getattr(arr, 'shape', None)}"
        )

    return arr.astype(np.float32)


def validate_input_schema(df_clean: pd.DataFrame, emb_matrix: np.ndarray) -> None:
    missing_c = [c for c in REQUIRED_CLEAN_COLS if c not in df_clean.columns]
    if missing_c:
        sys.exit(f"ERROR: Faltan columnas obligatorias en cleaned_companies.csv: {missing_c}")

    if df_clean["company_id"].duplicated().any():
        sys.exit("ERROR: company_id duplicado en cleaned_companies.csv; revise los datos.")

    if emb_matrix.shape[0] != len(df_clean):
        sys.exit(
            "ERROR: El número de embeddings no coincide con el número de filas del dataset limpio. "
            f"clean={len(df_clean)}, embeddings={emb_matrix.shape[0]}"
        )


def _pick_n_clusters(n_samples: int) -> int:
    if n_samples <= 0:
        return 0
    if n_samples == 1:
        return 1
    k = int(round(math.sqrt(n_samples)))
    k = max(5, min(28, k))
    return min(k, n_samples)


def generate_clustering(
    df: pd.DataFrame, embedding_matrix: np.ndarray
) -> tuple[np.ndarray, int]:
    """Returns cluster_id per row (same order as df), and n_clusters used."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int), 0
    if embedding_matrix.shape[0] != n:
        raise ValueError("embedding_matrix y dataframe tienen longitudes distintas")
    X = normalize(embedding_matrix, norm="l2", axis=1)
    k = _pick_n_clusters(n)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    return labels.astype(int), k


def _score_text_against_themes(
    text_f: str, themes: tuple[tuple[str, tuple[str, ...]], ...]
) -> list[tuple[str, int]]:
    scored: list[tuple[str, int]] = []
    for label, kws in themes:
        hits = sum(1 for kw in kws if kw in text_f)
        if hits:
            scored.append((label, hits))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def label_clusters(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> dict[int, str]:
    """Map cluster_id -> human-readable cluster_label."""
    labels_map: dict[int, str] = {}
    for cid in sorted(set(cluster_ids.tolist())):
        mask = cluster_ids == cid
        sub = df.loc[mask]
        parts: list[str] = []
        for _, row in sub.iterrows():
            parts.append(fold(str(row.get("semantic_text", "") or "")))
            parts.append(fold(str(row.get("description_clean", "") or "")))
            parts.append(fold(str(row.get("sector_primary", "") or "")))
        blob = " ".join(parts)
        ranked = _score_text_against_themes(blob, THEME_TAG_KEYWORDS)
        if ranked:
            best = ranked[0][0]
            if best == "General Technology" and len(ranked) > 1:
                best = ranked[1][0]
            labels_map[int(cid)] = best
        else:
            labels_map[int(cid)] = "General Technology"
    return labels_map


def build_theme_tags_for_row(
    text_f: str,
    cluster_label: str,
    max_tags: int = 6,
) -> list[str]:
    ranked = _score_text_against_themes(text_f, THEME_TAG_KEYWORDS)
    tags: list[str] = []
    if cluster_label and cluster_label not in tags:
        tags.append(cluster_label)
    for lab, _ in ranked:
        if lab not in tags and lab != "General Technology":
            tags.append(lab)
        if len(tags) >= max_tags:
            break
    if len(tags) == 0:
        tags = ["General Technology"]
    return tags[:max_tags]


def detect_priority_industries(
    sector_primary: str,
    sector_list: str,
    description: str,
    semantic: str,
    theme_tags: list[str],
) -> list[str]:
    blob = fold(
        f"{sector_primary} {sector_list} {description} {semantic} {' '.join(theme_tags)}"
    )
    found: list[str] = []
    for label, kws in PRIORITY_INDUSTRY_KEYWORDS:
        if any(kw in blob for kw in kws):
            found.append(label)
    return sorted(set(found))


def calc_industry_alignment_score(
    priority_labels: list[str],
    sector_primary: str,
    sector_list: str,
    description: str,
    semantic: str,
) -> tuple[float, str, str]:
    """Returns score 0-35, evidence text, confidence."""
    if not priority_labels:
        return 0.0, "Sin coincidencias con industrias prioritarias en sector ni texto.", "low"

    sp = fold(sector_primary or "")
    sl = fold(sector_list or "")
    dc = fold(description or "")
    sm = fold(semantic or "")
    sector_blob = f"{sp} {sl}"
    text_blob = f"{dc} {sm}"

    strong_sector: list[str] = []
    for lbl in priority_labels:
        kws = next(kws for name, kws in PRIORITY_INDUSTRY_KEYWORDS if name == lbl)
        if any(kw in sector_blob for kw in kws):
            strong_sector.append(lbl)

    semantic_hits = [lbl for lbl in priority_labels if lbl not in strong_sector]

    if strong_sector:
        n = len(set(strong_sector))
        if n >= 2:
            score = 35.0
            conf = "high"
        else:
            score = 32.0
            conf = "high"
        ev = f"Alineación directa en sector: {', '.join(sorted(set(strong_sector)))}."
        if semantic_hits:
            score = min(35.0, score + 1.0)
            ev += f" Refuerzo en descripción/semantic_text: {', '.join(semantic_hits)}."
        return float(min(35.0, score)), ev, conf

    inferred = list(priority_labels)
    if len(inferred) >= 2:
        score = 28.0
        conf = "medium"
        ev = f"Alineación inferida en descripción/semantic_text (múltiples): {', '.join(inferred)}."
        return float(min(35.0, score)), ev, conf

    lbl = inferred[0]
    kws = next(kws for name, kws in PRIORITY_INDUSTRY_KEYWORDS if name == lbl)
    hits = sum(1 for kw in kws if kw in text_blob)
    if hits >= 3:
        score = 30.0
        conf = "medium"
        ev = f"Alineación clara por texto fuerte con {lbl}."
    elif hits >= 1:
        score = 18.0
        conf = "medium"
        ev = f"Relación parcial por menciones de {lbl} en texto."
    else:
        score = 12.0
        conf = "low"
        ev = f"Señal débil o ambigua para {lbl}."
    return float(min(35.0, max(0.0, score))), ev, conf


def classify_business_impact(
    description: str,
    semantic: str,
    theme_tags: list[str],
    cluster_label: str,
) -> tuple[list[str], bool, bool, bool]:
    text_f = fold(f"{description} {semantic} {cluster_label} {' '.join(theme_tags)}")

    sell_kw = (
        "ventas",
        "vender",
        "ingresos",
        "revenue",
        "conversion",
        "comercial",
        "pipeline",
        "demanda",
        "crecimiento",
        "monetiz",
    )
    cx_kw = (
        "experiencia del cliente",
        "experiencia cliente",
        "customer experience",
        "cx",
        "satisfaccion",
        "fideliz",
        "personalizacion",
        "journey",
    )
    ops_kw = (
        "eficiencia",
        "costos",
        "optimiz",
        "automatiz",
        "procesos",
        "operaciones",
        "productividad",
        "reducir",
    )

    sell = any(k in text_f for k in sell_kw)
    cx = any(k in text_f for k in cx_kw)
    ops = any(k in text_f for k in ops_kw)

    labels: list[str] = []
    if sell:
        labels.append("sell_more")
    if cx:
        labels.append("customer_experience")
    if ops:
        labels.append("cost_process_optimization")

    return labels, sell, cx, ops


def calc_impact_alignment_score(
    impact_labels: list[str],
    description: str,
    semantic: str,
) -> tuple[float, str, str]:
    text_f = fold(f"{description} {semantic}")
    if not impact_labels:
        return 0.0, "No se detectaron señales claras de impacto en ventas, CX u operaciones.", "low"

    # Weight by keyword density per pillar
    def pillar_strength(kws: tuple[str, ...]) -> int:
        return sum(1 for k in kws if k in text_f)

    sell_h = pillar_strength(
        ("ventas", "vender", "ingresos", "conversion", "comercial", "crecimiento", "monetiz")
    )
    cx_h = pillar_strength(
        ("experiencia", "customer", "satisfaccion", "fideliz", "journey", "cx ")
    )
    ops_h = pillar_strength(
        ("eficiencia", "costos", "automatiz", "procesos", "operaciones", "optimiz")
    )

    strengths = [sell_h, cx_h, ops_h]
    active = sum(1 for x in strengths if x > 0)
    max_h = max(strengths)

    if max_h >= 4 and active >= 2:
        score = 35.0
        conf = "high"
        ev = "Impacto fuerte y explícito en múltiples frentes (texto denso en señales)."
    elif max_h >= 3:
        score = 30.0
        conf = "high"
        ev = "Impacto claro en al menos un pilar con buena densidad de términos."
    elif max_h >= 1 and active >= 2:
        score = 22.0
        conf = "medium"
        ev = "Impacto parcial en varios pilares con señal moderada."
    elif max_h >= 1:
        score = 14.0
        conf = "medium"
        ev = "Impacto secundario o dominante en un solo pilar."
    else:
        score = 6.0
        conf = "low"
        ev = "Impacto apenas sugerido por etiquetas; texto poco explícito."

    return float(min(35.0, score)), ev, conf


def calc_website_quality_score(
    website_clean: str,
    website_domain: str,
    invalid_website_flag: Any,
) -> float:
    inv = str(invalid_website_flag).strip().lower() in ("1", "true", "yes", "si")
    w = str(website_clean or "").strip()
    d = str(website_domain or "").strip()

    if inv or not w:
        return 0.0
    if len(w) < 6 or not re.search(r"[.]", w):
        return 3.0
    if d and re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", fold(d).replace(" ", "")):
        if len(w) >= 12 and w.lower().startswith(("http://", "https://")):
            return 10.0
        return 8.0
    return 5.0


def calc_description_quality_score(
    description_clean: str,
    missing_description_flag: Any,
    low_information_flag: Any,
) -> float:
    miss = str(missing_description_flag).strip().lower() in ("1", "true", "yes", "si")
    low = str(low_information_flag).strip().lower() in ("1", "true", "yes", "si")
    t = str(description_clean or "").strip()
    n = len(t)

    if miss or n == 0:
        return 0.0
    if low:
        return max(2.0, min(5.0, n / 80.0))
    if n >= 200 and t.count(" ") >= 25:
        return 10.0
    if n >= 100:
        return 7.0
    if n >= 40:
        return 4.0
    return 2.0


def calc_record_completeness_score(
    sector_primary: str,
    website_clean: str,
    description_clean: str,
    semantic_text: str,
    missing_description_flag: Any,
    invalid_website_flag: Any,
    low_information_flag: Any,
) -> float:
    pts = 0.0
    if str(sector_primary or "").strip():
        pts += 2.0
    if str(website_clean or "").strip() and str(invalid_website_flag).lower() not in ("1", "true"):
        pts += 2.5
    if str(description_clean or "").strip() and str(missing_description_flag).lower() not in ("1", "true"):
        pts += 2.5
    st = str(semantic_text or "").strip()
    if len(st) >= 120:
        pts += 2.5
    elif len(st) >= 40:
        pts += 1.5
    if str(low_information_flag).lower() not in ("1", "true"):
        pts += 0.5
    return float(min(10.0, round(pts, 1)))


def calc_data_maturity_score(
    website_quality_score: float,
    description_quality_score: float,
    record_completeness_score: float,
) -> tuple[float, str]:
    # Direct sum: each 0-10 -> total 0-30
    total = website_quality_score + description_quality_score + record_completeness_score
    total = float(min(30.0, round(total, 1)))
    ev = (
        f"website={website_quality_score:.1f}/10, "
        f"descripcion={description_quality_score:.1f}/10, "
        f"completitud={record_completeness_score:.1f}/10 (suma acotada a 30)."
    )
    return total, ev


def calc_radar_score_total(
    industry_alignment_score: float,
    impact_alignment_score: float,
    data_maturity_score: float,
) -> float:
    s = industry_alignment_score + impact_alignment_score + data_maturity_score
    s = max(0.0, min(100.0, float(s)))
    return round(s, 1)


def assign_priority_label(radar_score: float) -> str:
    if radar_score > 80:
        return "CANDIDATE"
    if radar_score >= 40:
        return "REVIEW"
    return "DISCARD"


def build_score_explanation(
    radar: float,
    industry_score: float,
    impact_score: float,
    maturity_score: float,
    industry_ev: str,
    impact_ev: str,
    maturity_ev: str,
    priority_industries: list[str],
    impact_labels: list[str],
) -> tuple[str, str, str]:
    summary = (
        f"Radar {radar:.1f}/100: "
        f"industria {industry_score:.1f}/35, impacto {impact_score:.1f}/35, "
        f"madurez de datos {maturity_score:.1f}/30. "
        f"Industrias: {', '.join(priority_industries) or 'ninguna prioritaria'}. "
        f"Impacto detectado: {', '.join(impact_labels) or 'ninguno'}."
    )
    strengths: list[str] = []
    weaknesses: list[str] = []
    if industry_score >= 25:
        strengths.append(f"Alineación industrial ({industry_score:.1f}/35)")
    elif industry_score < 10:
        weaknesses.append("Alineación industrial débil")
    if impact_score >= 25:
        strengths.append(f"Impacto de negocio claro ({impact_score:.1f}/35)")
    elif impact_score < 10:
        weaknesses.append("Impacto de negocio poco evidenciado")
    if maturity_score >= 20:
        strengths.append(f"Buena madurez de datos ({maturity_score:.1f}/30)")
    elif maturity_score < 12:
        weaknesses.append("Datos del registro incompletos o de baja calidad")

    top_s = " | ".join(strengths[:3]) if strengths else "Sin fortalezas destacadas en sub-scores"
    top_w = " | ".join(weaknesses[:3]) if weaknesses else "Sin debilidades destacadas en sub-scores"
    detail = f"Industria: {industry_ev} Impacto: {impact_ev} Madurez: {maturity_ev}"
    return summary + " " + detail, top_s, top_w


def export_outputs(
    df_csv_ready: pd.DataFrame,
    report: dict[str, Any],
    cluster_summary: pd.DataFrame,
    paths: tuple[Path, Path, Path],
) -> None:
    csv_p, js_p, cl_p = paths
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    df_csv = df_csv_ready.copy()
    for col in ("theme_tags", "priority_industry_labels", "impact_labels"):
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )

    df_csv.to_csv(csv_p, index=False, encoding="utf-8")
    js_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_summary.to_csv(cl_p, index=False, encoding="utf-8")

def generate_report_payload(
    total_rows_input: int,
    total_rows_scored: int,
    total_rows_skipped: int,
    df_scored: pd.DataFrame,
    embedding_provider: embedding_provider,
    embedding_model: embedding_model,
    ts: str,
) -> dict[str, Any]:
    return {
        "total_rows_input": total_rows_input,
        "total_rows_scored": total_rows_scored,
        "total_rows_skipped": total_rows_skipped,
        "total_candidate": int((df_scored["priority_label"] == "CANDIDATE").sum()),
        "total_review": int((df_scored["priority_label"] == "REVIEW").sum()),
        "total_discard": int((df_scored["priority_label"] == "DISCARD").sum()),
        "avg_radar_score": float(round(df_scored["radar_score"].mean(), 4)) if len(df_scored) else 0.0,
        "avg_industry_alignment_score": float(round(df_scored["industry_alignment_score"].mean(), 4))
        if len(df_scored)
        else 0.0,
        "avg_impact_alignment_score": float(round(df_scored["impact_alignment_score"].mean(), 4))
        if len(df_scored)
        else 0.0,
        "avg_data_maturity_score": float(round(df_scored["data_maturity_score"].mean(), 4))
        if len(df_scored)
        else 0.0,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "execution_timestamp": ts,
    }


def _eligible_row(row: pd.Series) -> bool:
    cid = row.get("company_id")
    if cid is None:
        return False
    try:
        if pd.isna(cid):
            return False
    except (TypeError, ValueError):
        return False
    if str(cid).strip() == "":
        return False

    rs = str(row.get("record_status", "")).strip()
    if rs not in VALID_RECORD_STATUS:
        return False

    return True
