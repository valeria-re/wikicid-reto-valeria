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
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_CSV = OUTPUT_DIR / "companies_clean.csv"
EMB_PARQUET = OUTPUT_DIR / "companies_embeddings.parquet"
OUT_CSV = OUTPUT_DIR / "companies_scored.csv"
OUT_PARQUET = OUTPUT_DIR / "companies_scored.parquet"
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

REQUIRED_EMB_COLS = (
    "company_id",
    "embedding_vector",
    "embedding_dim",
    "embedding_provider",
    "embedding_model",
    "embedding_status",
    "semantic_text_used",
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


def load_embeddings(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(
            f"ERROR: No existe el archivo de embeddings requerido: {path}. "
            "Ejecute primero el pipeline de embeddings."
        )
    return pd.read_parquet(path)


def validate_input_schema(df_clean: pd.DataFrame, df_emb: pd.DataFrame) -> None:
    missing_c = [c for c in REQUIRED_CLEAN_COLS if c not in df_clean.columns]
    if missing_c:
        sys.exit(f"ERROR: Faltan columnas obligatorias en companies_clean.csv: {missing_c}")
    missing_e = [c for c in REQUIRED_EMB_COLS if c not in df_emb.columns]
    if missing_e:
        sys.exit(f"ERROR: Faltan columnas obligatorias en companies_embeddings.parquet: {missing_e}")
    if df_clean["company_id"].duplicated().any():
        sys.exit("ERROR: company_id duplicado en companies_clean.csv; revise los datos.")
    if df_emb["company_id"].duplicated().any():
        sys.exit("ERROR: company_id duplicado en companies_embeddings.parquet; revise los datos.")


def merge_datasets(df_clean: pd.DataFrame, df_emb: pd.DataFrame) -> pd.DataFrame:
    emb_keep = list(REQUIRED_EMB_COLS)
    sub = df_emb[emb_keep].copy()
    merged = df_clean.merge(sub, on="company_id", how="inner", validate="one_to_one")
    return merged


def _to_embedding_matrix(vectors: Any) -> np.ndarray:
    """Stack embedding_vector column into (n, dim) float32."""
    rows: list[np.ndarray] = []
    for v in vectors:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            rows.append(np.array([], dtype=np.float32))
            continue
        arr = np.asarray(v, dtype=np.float32).ravel()
        rows.append(arr)
    lens = {x.shape[0] for x in rows if x.size}
    if len(lens) > 1:
        raise ValueError("embedding_vector: dimensiones inconsistentes entre filas")
    if not lens:
        return np.zeros((0, 0), dtype=np.float32)
    dim = lens.pop()
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r.size == dim:
            out[i] = r
    return out


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
    df_for_parquet: pd.DataFrame,
    report: dict[str, Any],
    cluster_summary: pd.DataFrame,
    paths: tuple[Path, Path, Path, Path],
) -> None:
    """Parquet conserva listas; CSV serializa listas como JSON."""
    csv_p, pq_p, js_p, cl_p = paths
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df_csv = df_for_parquet.copy()
    for col in ("theme_tags", "priority_industry_labels", "impact_labels"):
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )
    df_csv.to_csv(csv_p, index=False)
    df_for_parquet.to_parquet(pq_p, index=False)
    js_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_summary.to_csv(cl_p, index=False)


def generate_report_payload(
    total_rows_input: int,
    total_rows_scored: int,
    total_rows_skipped: int,
    df_scored: pd.DataFrame,
    embedding_provider: str,
    embedding_model: str,
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
    if str(row.get("embedding_status", "")).strip() != "embedded":
        return False
    ev = row.get("embedding_vector")
    if ev is None:
        return False
    try:
        a = np.asarray(ev, dtype=np.float32).ravel()
        if a.size == 0:
            return False
    except (TypeError, ValueError):
        return False
    return True


def run_scoring() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    df_clean = load_clean_dataset(CLEAN_CSV)
    df_emb = load_embeddings(EMB_PARQUET)
    validate_input_schema(df_clean, df_emb)

    total_rows_input = len(df_clean)
    merged = merge_datasets(df_clean, df_emb)

    eligible_mask = merged.apply(_eligible_row, axis=1)
    eligible_idx = merged.index[eligible_mask]
    skipped = int((~eligible_mask).sum())

    emb_provider = ""
    emb_model = ""

    # Initialize output columns on merged copy
    out = merged.copy()
    float_cols = (
        "industry_alignment_score",
        "impact_alignment_score",
        "website_quality_score",
        "description_quality_score",
        "record_completeness_score",
        "data_maturity_score",
        "radar_score",
    )
    str_cols = (
        "cluster_label",
        "industry_alignment_evidence",
        "industry_alignment_confidence",
        "impact_alignment_evidence",
        "impact_alignment_confidence",
        "data_maturity_evidence",
        "priority_label",
        "score_reason_summary",
        "top_strengths",
        "top_weaknesses",
    )
    list_cols = ("theme_tags", "priority_industry_labels", "impact_labels")
    bool_cols = (
        "impact_sell_more_flag",
        "impact_customer_experience_flag",
        "impact_cost_process_flag",
    )
    for c in float_cols:
        out[c] = np.nan
    out["cluster_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for c in str_cols:
        out[c] = pd.Series([pd.NA] * len(out), dtype=object)
    for c in list_cols:
        out[c] = pd.Series([None] * len(out), dtype=object)
    for c in bool_cols:
        out[c] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    n_eligible = int(eligible_mask.sum())
    cluster_count = 0

    if n_eligible == 0:
        df_scored = out.iloc[0:0].copy()
        if len(merged):
            emb_provider = str(merged["embedding_provider"].iloc[0])
            emb_model = str(merged["embedding_model"].iloc[0])
    else:
        sub = merged.loc[eligible_idx].copy()
        emb_mat = _to_embedding_matrix(sub["embedding_vector"].tolist())
        cluster_ids, cluster_count = generate_clustering(sub, emb_mat)
        id_to_label = label_clusters(sub, cluster_ids)

        for j, ix in enumerate(sub.index):
            cid = int(cluster_ids[j])
            clab = id_to_label.get(cid, "General Technology")
            row = sub.loc[ix]
            sem = str(row.get("semantic_text", "") or row.get("semantic_text_used", "") or "")
            text_f = fold(f"{sem} {row.get('description_clean', '')} {row.get('sector_primary', '')}")
            tags = build_theme_tags_for_row(text_f, clab)

            pri = detect_priority_industries(
                str(row.get("sector_primary", "")),
                str(row.get("sector_list_clean", "")),
                str(row.get("description_clean", "")),
                sem,
                tags,
            )
            ind_s, ind_ev, ind_cf = calc_industry_alignment_score(
                pri,
                str(row.get("sector_primary", "")),
                str(row.get("sector_list_clean", "")),
                str(row.get("description_clean", "")),
                sem,
            )

            imp_lbls, f_sell, f_cx, f_ops = classify_business_impact(
                str(row.get("description_clean", "")),
                sem,
                tags,
                clab,
            )
            imp_s, imp_ev, imp_cf = calc_impact_alignment_score(
                imp_lbls,
                str(row.get("description_clean", "")),
                sem,
            )

            wq = calc_website_quality_score(
                str(row.get("website_clean", "")),
                str(row.get("website_domain", "")),
                row.get("invalid_website_flag"),
            )
            dq = calc_description_quality_score(
                str(row.get("description_clean", "")),
                row.get("missing_description_flag"),
                row.get("low_information_flag"),
            )
            rq = calc_record_completeness_score(
                str(row.get("sector_primary", "")),
                str(row.get("website_clean", "")),
                str(row.get("description_clean", "")),
                str(row.get("semantic_text", "")),
                row.get("missing_description_flag"),
                row.get("invalid_website_flag"),
                row.get("low_information_flag"),
            )
            dm, dm_ev = calc_data_maturity_score(wq, dq, rq)
            radar = calc_radar_score_total(ind_s, imp_s, dm)
            plab = assign_priority_label(radar)
            summ, ts_s, ts_w = build_score_explanation(
                radar,
                ind_s,
                imp_s,
                dm,
                ind_ev,
                imp_ev,
                dm_ev,
                pri,
                imp_lbls,
            )

            out.at[ix, "cluster_id"] = cid
            out.at[ix, "cluster_label"] = clab
            out.at[ix, "theme_tags"] = tags
            out.at[ix, "priority_industry_labels"] = pri
            out.at[ix, "industry_alignment_score"] = ind_s
            out.at[ix, "industry_alignment_evidence"] = ind_ev
            out.at[ix, "industry_alignment_confidence"] = ind_cf
            out.at[ix, "impact_labels"] = imp_lbls
            out.at[ix, "impact_sell_more_flag"] = f_sell
            out.at[ix, "impact_customer_experience_flag"] = f_cx
            out.at[ix, "impact_cost_process_flag"] = f_ops
            out.at[ix, "impact_alignment_score"] = imp_s
            out.at[ix, "impact_alignment_evidence"] = imp_ev
            out.at[ix, "impact_alignment_confidence"] = imp_cf
            out.at[ix, "website_quality_score"] = wq
            out.at[ix, "description_quality_score"] = dq
            out.at[ix, "record_completeness_score"] = rq
            out.at[ix, "data_maturity_score"] = dm
            out.at[ix, "data_maturity_evidence"] = dm_ev
            out.at[ix, "radar_score"] = radar
            out.at[ix, "priority_label"] = plab
            out.at[ix, "score_reason_summary"] = summ
            out.at[ix, "top_strengths"] = ts_s
            out.at[ix, "top_weaknesses"] = ts_w

        df_scored = out.loc[eligible_idx].copy()
        emb_provider = str(df_scored["embedding_provider"].iloc[0])
        emb_model = str(df_scored["embedding_model"].iloc[0])

    report = generate_report_payload(
        total_rows_input,
        n_eligible,
        skipped + (total_rows_input - len(merged)),
        df_scored,
        emb_provider,
        emb_model,
        ts,
    )

    # cluster_summary from scored rows only
    if len(df_scored) and "cluster_id" in df_scored.columns:
        def agg_tags(series: pd.Series) -> str:
            all_tags: list[str] = []
            for v in series:
                if isinstance(v, list):
                    all_tags.extend(v)
            freq: dict[str, int] = {}
            for t in all_tags:
                freq[t] = freq.get(t, 0) + 1
            top = sorted(freq.items(), key=lambda x: -x[1])[:8]
            return "; ".join(f"{k}" for k, _ in top)

        cs = (
            df_scored.groupby(["cluster_id", "cluster_label"], dropna=False)
            .agg(
                total_companies=("company_id", "count"),
                representative_theme_tags=("theme_tags", agg_tags),
                average_radar_score=("radar_score", "mean"),
            )
            .reset_index()
        )
    else:
        cs = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "total_companies",
                "representative_theme_tags",
                "average_radar_score",
            ]
        )

    export_outputs(
        out,
        report,
        cs,
        (OUT_CSV, OUT_PARQUET, REPORT_JSON, CLUSTER_SUMMARY_CSV),
    )

    avg_rs = float(df_scored["radar_score"].mean()) if len(df_scored) else 0.0
    print("--- Resumen scoring ---")
    print(f"Filas leídas (clean): {total_rows_input}")
    print(f"Filas tras merge con embeddings: {len(merged)}")
    print(f"Filas con embedding elegible (válidas + embedded): {n_eligible}")
    print(f"Filas saltadas (no elegibles o sin par en embeddings): {report['total_rows_skipped']}")
    print(f"Clusters generados: {cluster_count}")
    print(f"Total CANDIDATE: {report['total_candidate']}")
    print(f"Total REVIEW: {report['total_review']}")
    print(f"Total DISCARD: {report['total_discard']}")
    print(f"Score promedio (radar): {avg_rs:.2f}")
    print(
        "Archivos generados:",
        f"{OUT_CSV.name}, {OUT_PARQUET.name}, {REPORT_JSON.name}, {CLUSTER_SUMMARY_CSV.name}",
    )


def main() -> None:
    run_scoring()


if __name__ == "__main__":
    main()
