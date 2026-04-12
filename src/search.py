"""
Búsqueda semántica sobre empresas ya puntuadas y embebidas (sin cleaning ni re-embedding de empresas).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"

# Columnas de embedding que pueden venir duplicadas en companies_scored; prevalece companies_embeddings.parquet.
_SCORED_DROP_IF_PRESENT = (
    "embedding_vector",
    "embedding_dim",
    "embedding_provider",
    "embedding_model",
    "embedding_status",
    "semantic_text_used",
)

_SCORED_CSV = _DEFAULT_OUTPUT_DIR / "companies_scored.csv"
_SCORED_PARQUET = _DEFAULT_OUTPUT_DIR / "companies_scored.parquet"
_EMBEDDINGS_PARQUET = _DEFAULT_OUTPUT_DIR / "companies_embeddings.parquet"

REQUIRED_SCORED_COLS: tuple[str, ...] = (
    "company_id",
    "company_name_display",
    "company_name_normalized",
    "sector_primary",
    "website_domain",
    "cluster_id",
    "cluster_label",
    "theme_tags",
    "priority_industry_labels",
    "industry_alignment_score",
    "impact_alignment_score",
    "data_maturity_score",
    "radar_score",
    "priority_label",
    "score_reason_summary",
    "record_status",
)

REQUIRED_EMBEDDING_COLS: tuple[str, ...] = (
    "company_id",
    "embedding_vector",
    "embedding_dim",
    "embedding_provider",
    "embedding_model",
    "embedding_status",
    "semantic_text_used",
)

VALID_PRIORITY_LABELS = frozenset({"CANDIDATE", "REVIEW", "DISCARD"})
VALID_RECORD_STATUS = frozenset({"valid_record", "partial_record"})

_RANKING_FORMULA_NAME = "hybrid_semantic_dominant_gated_radar_v1"
_DEFAULT_TOP_K = 20

_MULTI_SPACE = re.compile(r"\s+")


class SearchAbort(Exception):
    """Error de validación o datos; mensaje listo para consola."""


@dataclass
class SearchParams:
    query: str
    top_k: int = _DEFAULT_TOP_K
    score_threshold: float | None = None
    similarity_threshold: float | None = None
    allowed_priority_labels: list[str] | None = None
    allowed_sectors: list[str] | None = None
    allowed_clusters: list[str] | None = None
    exclude_discard: bool = False
    export_results: bool = False


@dataclass
class FilterLog:
    step: str
    rows_in: int
    rows_out: int
    note: str = ""


@dataclass
class SearchContext:
    filter_logs: list[FilterLog] = field(default_factory=list)

    def log_filter(self, step: str, rows_in: int, rows_out: int, note: str = "") -> None:
        self.filter_logs.append(FilterLog(step, rows_in, rows_out, note))


def _resolve_scored_path() -> Path:
    if _SCORED_PARQUET.is_file():
        return _SCORED_PARQUET
    if _SCORED_CSV.is_file():
        return _SCORED_CSV
    raise SearchAbort(
        "Falta el dataset puntuado: se esperaba "
        f"`{_SCORED_PARQUET.name}` o `{_SCORED_CSV.name}` en `{_DEFAULT_OUTPUT_DIR}`."
    )


def load_scored_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_embeddings_parquet(path: Path = _EMBEDDINGS_PARQUET) -> pd.DataFrame:
    if not path.is_file():
        raise SearchAbort(
            f"Falta el archivo de embeddings: `{path}` (ruta absoluta esperada o bajo outputs/)."
        )
    return pd.read_parquet(path)


def load_embeddings_or_from_scored(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Usa `companies_embeddings.parquet` si existe; si no, toma las columnas de embedding
    ya presentes en el dataset puntuado (p. ej. `companies_scored.csv` unificado).
    """
    if _EMBEDDINGS_PARQUET.is_file():
        return load_embeddings_parquet(_EMBEDDINGS_PARQUET)
    cols = list(REQUIRED_EMBEDDING_COLS)
    miss = [c for c in cols if c not in scored.columns]
    if miss:
        raise SearchAbort(
            f"No existe `{_EMBEDDINGS_PARQUET.name}` en `{_DEFAULT_OUTPUT_DIR}` y faltan "
            f"columnas de embedding en el dataset puntuado: {miss}."
        )
    return scored[cols].copy()


def validate_input_schema(
    scored: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> None:
    miss_s = [c for c in REQUIRED_SCORED_COLS if c not in scored.columns]
    if miss_s:
        raise SearchAbort(
            f"Faltan columnas obligatorias en el dataset puntuado: {miss_s}. "
            f"Columnas presentes: {list(scored.columns)}"
        )
    miss_e = [c for c in REQUIRED_EMBEDDING_COLS if c not in embeddings.columns]
    if miss_e:
        raise SearchAbort(
            f"Faltan columnas obligatorias en embeddings: {miss_e}. "
            f"Columnas presentes: {list(embeddings.columns)}"
        )


def _assert_no_duplicate_company_ids(df: pd.DataFrame, name: str) -> None:
    dup = df["company_id"].astype(str)
    if dup.duplicated().any():
        n = int(dup.duplicated().sum())
        raise SearchAbort(
            f"Hay {n} `company_id` duplicados en {name}; la unión 1:1 exige ids únicos."
        )


def merge_scored_and_embeddings(
    scored: pd.DataFrame,
    embeddings: pd.DataFrame,
    ctx: SearchContext,
) -> pd.DataFrame:
    _assert_no_duplicate_company_ids(scored, "companies_scored")
    _assert_no_duplicate_company_ids(embeddings, "companies_embeddings")

    drop_cols = [c for c in _SCORED_DROP_IF_PRESENT if c in scored.columns]
    scored = scored.drop(columns=drop_cols, errors="ignore")

    emb_cols = [
        "company_id",
        "embedding_vector",
        "embedding_dim",
        "embedding_provider",
        "embedding_model",
        "embedding_status",
        "semantic_text_used",
    ]
    missing_emb = [c for c in emb_cols if c not in embeddings.columns]
    if missing_emb:
        raise SearchAbort(f"Faltan columnas en embeddings para el merge: {missing_emb}")
    emb = embeddings[emb_cols].copy()
    emb["company_id"] = emb["company_id"].astype(str)
    sc = scored.copy()
    sc["company_id"] = sc["company_id"].astype(str)

    merged = sc.merge(emb, on="company_id", how="inner", suffixes=("_scored", "_emb"))

    rows_in = len(merged)
    merged = merged.loc[merged["embedding_status"].astype(str).str.strip() == "embedded"].copy()
    ctx.log_filter("embedding_status == embedded", rows_in, len(merged))

    rows_in = len(merged)
    rs = merged["record_status"].astype(str).str.strip()
    merged = merged.loc[rs.isin(VALID_RECORD_STATUS)].copy()
    ctx.log_filter("record_status in {valid_record, partial_record}", rows_in, len(merged))

    return merged


def validate_and_prepare_query(raw: str) -> str:
    if raw is None or not str(raw).strip():
        raise SearchAbort("La consulta `query` no puede ser nula ni vacía.")
    t = str(raw).strip()
    t = _MULTI_SPACE.sub(" ", t).strip()
    if not t:
        raise SearchAbort("La consulta `query` no puede ser nula ni vacía.")
    return t


def validate_search_params(p: SearchParams) -> None:
    if not p.query or not str(p.query).strip():
        raise SearchAbort("La consulta `query` no puede ser nula ni vacía.")
    if p.top_k is not None and (not isinstance(p.top_k, int) or p.top_k < 1):
        raise SearchAbort("`top_k` debe ser un entero positivo.")
    if p.score_threshold is not None:
        if not (0 <= float(p.score_threshold) <= 100):
            raise SearchAbort("`score_threshold` debe estar entre 0 y 100.")
    if p.similarity_threshold is not None:
        if not (0 <= float(p.similarity_threshold) <= 1):
            raise SearchAbort("`similarity_threshold` debe estar entre 0 y 1.")
    if p.allowed_priority_labels:
        bad = [x for x in p.allowed_priority_labels if x not in VALID_PRIORITY_LABELS]
        if bad:
            raise SearchAbort(
                f"`allowed_priority_labels` contiene valores inválidos {bad}. "
                f"Permitidos: {sorted(VALID_PRIORITY_LABELS)}."
            )


def _unique_provider_model(embeddings: pd.DataFrame) -> tuple[str, str]:
    prov = embeddings["embedding_provider"].astype(str).unique().tolist()
    mod = embeddings["embedding_model"].astype(str).unique().tolist()
    if len(prov) != 1 or len(mod) != 1:
        raise SearchAbort(
            "Inconsistencia: se requiere un único `embedding_provider` y un único "
            f"`embedding_model` en companies_embeddings; encontrado providers={prov!r} models={mod!r}."
        )
    return str(prov[0]), str(mod[0])


def get_embedding_backend_for_query(provider: str, model: str):
    """Cliente/proveedor para embeder la query (mismo proveedor/modelo que el parquet)."""
    import embeddings as emb

    cfg = emb.EmbeddingConfig(
        provider=provider,
        model=model,
        batch_size=32,
        max_text_length=8000,
        min_text_length=1,
        max_retries=3,
        input_path=_REPO_ROOT / "outputs" / "companies_clean.csv",
        output_dir=_DEFAULT_OUTPUT_DIR,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        incremental=False,
    )
    if str(provider).lower() == "openai":
        raise SearchAbort(
            "Este script no realiza llamadas a red (política sin internet). "
            "Las consultas con embeddings `openai` no están soportadas aquí; "
            "use un `companies_embeddings.parquet` generado con proveedor `local`."
        )
    return emb.get_embedding_backend(cfg), cfg


def generate_query_embedding(
    query_text: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    import embeddings as emb

    backend, cfg = get_embedding_backend_for_query(provider, model)
    arr, err = emb.embed_single_with_retries(backend, query_text, cfg)
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    if arr is None:
        raise SearchAbort(f"No se pudo generar el embedding de la consulta: {err or 'error desconocido'}")

    vec = np.asarray(arr, dtype=np.float64).reshape(-1)
    dim = int(vec.shape[0])
    return {
        "query_text": query_text,
        "query_embedding": vec,
        "query_embedding_dim": dim,
        "query_embedding_provider": provider,
        "query_embedding_model": model,
    }


def _vector_to_1d(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).reshape(-1)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float64).reshape(-1)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return np.zeros(0, dtype=np.float64)
        # CSV a menudo guarda el repr de numpy: "[ -0.02 ... ]" con saltos de línea
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].replace("\n", " ")
            try:
                return np.fromstring(inner, sep=" ", dtype=np.float64).reshape(-1)
            except ValueError:
                pass
        try:
            parsed = json.loads(s)
            return np.asarray(parsed, dtype=np.float64).reshape(-1)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return np.asarray(v, dtype=np.float64).reshape(-1)


def cosine_similarity_matrix(
    company_matrix: np.ndarray,
    query_vec: np.ndarray,
) -> np.ndarray:
    """
    Similitud coseno fila a fila. Vectores de empresa y query se normalizan L2.
    Tras cos_raw en [-1, 1], se mapea a semantic_similarity en [0, 1] con (cos_raw + 1) / 2.
    """
    x = np.asarray(company_matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    q = np.asarray(query_vec, dtype=np.float64).reshape(-1)

    if x.shape[1] != q.shape[0]:
        raise SearchAbort(
            f"Dimensión de embedding incompatible: empresas d={x.shape[1]}, consulta d={q.shape[0]}."
        )

    row_norms = np.linalg.norm(x, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    x_unit = x / row_norms

    q_norm = float(np.linalg.norm(q))
    q_unit = q / max(q_norm, 1e-12)

    cos_raw = (x_unit @ q_unit).ravel()
    # Mapeo a [0, 1] para umbral comparable (spec: similarity_threshold 0..1)
    semantic_01 = (cos_raw + 1.0) / 2.0
    return np.clip(semantic_01, 0.0, 1.0)


def apply_filters(
    df: pd.DataFrame,
    params: SearchParams,
    ctx: SearchContext,
) -> pd.DataFrame:
    out = df
    rows_in = len(out)

    if params.score_threshold is not None:
        thr = float(params.score_threshold)
        rs = pd.to_numeric(out["radar_score"], errors="coerce").fillna(-np.inf)
        out = out.loc[rs >= thr].copy()
        ctx.log_filter(f"radar_score >= {thr}", rows_in, len(out))
        rows_in = len(out)

    if params.similarity_threshold is not None:
        thr = float(params.similarity_threshold)
        out = out.loc[out["semantic_similarity"] >= thr].copy()
        ctx.log_filter(f"semantic_similarity >= {thr}", rows_in, len(out))
        rows_in = len(out)

    if params.allowed_priority_labels:
        allowed = set(params.allowed_priority_labels)
        out = out.loc[out["priority_label"].astype(str).isin(allowed)].copy()
        ctx.log_filter(f"priority_label in {sorted(allowed)}", rows_in, len(out))
        rows_in = len(out)

    if params.exclude_discard:
        out = out.loc[out["priority_label"].astype(str).str.strip() != "DISCARD"].copy()
        ctx.log_filter("exclude_discard (priority_label != DISCARD)", rows_in, len(out))
        rows_in = len(out)

    if params.allowed_sectors:
        sectors = [s.strip().lower() for s in params.allowed_sectors if str(s).strip()]

        def sector_match(row: pd.Series) -> bool:
            sp = str(row.get("sector_primary", "")).lower()
            pil = str(row.get("priority_industry_labels", "")).lower()
            for s in sectors:
                if s in sp or s in pil:
                    return True
            return False

        mask = out.apply(sector_match, axis=1)
        out = out.loc[mask].copy()
        ctx.log_filter(f"allowed_sectors {params.allowed_sectors}", rows_in, len(out))
        rows_in = len(out)

    if params.allowed_clusters:
        cl = {str(c).strip().lower() for c in params.allowed_clusters if str(c).strip()}
        lab = out["cluster_label"].astype(str).str.strip().str.lower()
        out = out.loc[lab.isin(cl)].copy()
        ctx.log_filter(f"allowed_clusters {params.allowed_clusters}", rows_in, len(out))

    return out


def compute_hybrid_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking híbrido explícito (fórmula documentada en comentarios).

    - La similitud semántica (semantic_similarity en [0,1]) tiene peso dominante.
    - El radar y las alineaciones se ponderan y se multiplican por la similitud base
      para que un radar alto no eleve empresas sin encaje semántico fuerte.
    - Se añade un pequeño término según priority_label (CANDIDATE > REVIEW > DISCARD)
      acotado por similitud, para priorización estratégica sin tapar la relevancia.
    """
    out = df.copy()
    sem = pd.to_numeric(out["semantic_similarity"], errors="coerce").fillna(0.0).to_numpy()
    radar = pd.to_numeric(out["radar_score"], errors="coerce").fillna(0.0).to_numpy()
    ind = pd.to_numeric(out["industry_alignment_score"], errors="coerce").fillna(0.0).to_numpy()
    imp = pd.to_numeric(out["impact_alignment_score"], errors="coerce").fillna(0.0).to_numpy()
    dm = pd.to_numeric(out["data_maturity_score"], errors="coerce").fillna(0.0).to_numpy()

    radar_n = np.clip(radar / 100.0, 0.0, 1.0)
    ind_n = np.clip(ind / 100.0, 0.0, 1.0)
    imp_n = np.clip(imp / 100.0, 0.0, 1.0)
    dm_n = np.clip(dm / 100.0, 0.0, 1.0)

    prio_map = {"CANDIDATE": 1.0, "REVIEW": 0.5, "DISCARD": 0.0}
    pl = out["priority_label"].astype(str).str.strip().map(lambda x: prio_map.get(x, 0.0)).to_numpy()

    # Pesos (suma conceptual ~1 en componentes principales)
    w_sem = 0.56
    w_radar = 0.22
    w_ind = 0.08
    w_imp = 0.07
    w_dm = 0.05
    w_prio = 0.02

    # Términos estratégicos acotados por sem: evita que radar opaque a empresas relevantes
    # y evita que radar salve empresas irrelevantes.
    gate = sem
    ranking_score = (
        w_sem * sem
        + w_radar * radar_n * gate
        + w_ind * ind_n * gate
        + w_imp * imp_n * gate
        + w_dm * dm_n * gate
        + w_prio * pl * gate
    )

    out["ranking_score"] = ranking_score
    out = out.sort_values("ranking_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["result_rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def _safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def build_match_explanations(row: pd.Series) -> tuple[str, str, str]:
    sector = _safe_str(row.get("sector_primary"))
    cluster = _safe_str(row.get("cluster_label"))
    themes = _safe_str(row.get("theme_tags"))
    srs = _safe_str(row.get("score_reason_summary"))
    plab = _safe_str(row.get("priority_label"))
    rscore = row.get("radar_score")
    try:
        rnum = float(rscore)
    except (TypeError, ValueError):
        rnum = float("nan")

    semantic_match_reason = (
        f"Alineación temática con la consulta según sector «{sector}», "
        f"agrupación «{cluster}»"
        + (f" y etiquetas de tema «{themes}»." if themes else ".")
    )

    if not np.isnan(rnum):
        base = f"Prioridad «{plab}» con Radar Score {rnum:.1f}/100."
        strategic_match_reason = base + (f" Detalle registrado: {srs}" if srs else "")
    else:
        strategic_match_reason = (
            f"Prioridad «{plab}». Detalle registrado: {srs}" if srs else f"Prioridad «{plab}»."
        )

    match_reason_summary = (
        f"Encaje semántico con contexto de negocio ({sector}, {cluster}); "
        f"posición estratégica {plab} (radar {rnum:.1f}/100)." if not np.isnan(rnum) else
        f"Encaje semántico con contexto de negocio ({sector}, {cluster}); prioridad {plab}."
    )

    return semantic_match_reason, strategic_match_reason, match_reason_summary


def build_all_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sem_l: list[str] = []
    strat_l: list[str] = []
    sum_l: list[str] = []
    for _, row in out.iterrows():
        a, b, c = build_match_explanations(row)
        sem_l.append(a)
        strat_l.append(b)
        sum_l.append(c)
    out["semantic_match_reason"] = sem_l
    out["strategic_match_reason"] = strat_l
    out["match_reason_summary"] = sum_l
    return out


RESULT_COLUMNS: tuple[str, ...] = (
    "result_rank",
    "company_id",
    "company_name_display",
    "semantic_similarity",
    "ranking_score",
    "radar_score",
    "priority_label",
    "sector_primary",
    "cluster_label",
    "theme_tags",
    "priority_industry_labels",
    "industry_alignment_score",
    "impact_alignment_score",
    "data_maturity_score",
    "website_domain",
    "description_clean",
    "score_reason_summary",
    "semantic_match_reason",
    "strategic_match_reason",
    "match_reason_summary",
)


def format_final_results(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    return df[cols].copy()


def export_results(
    df: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "semantic_search_results.csv"
    pq_p = out_dir / "semantic_search_results.parquet"
    df.to_csv(csv_p, index=False)
    df.to_parquet(pq_p, index=False)
    return csv_p, pq_p


def generate_report_json(
    payload: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "semantic_search_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _export_if_requested(
    params: SearchParams,
    out_dir: Path,
    final_df: pd.DataFrame,
    report: dict[str, Any],
) -> list[str]:
    if not params.export_results:
        return []
    csv_p, pq_p = export_results(final_df, out_dir)
    js_p = generate_report_json(report, out_dir)
    return [str(csv_p), str(pq_p), str(js_p)]


def _validate_embedding_matrix_vs_meta(merged: pd.DataFrame, vectors: np.ndarray) -> None:
    dims = vectors.shape[1]
    declared = pd.to_numeric(merged["embedding_dim"], errors="coerce")
    if declared.notna().any():
        mode_v = int(declared.dropna().mode().iloc[0])
        if mode_v != dims:
            raise SearchAbort(
                f"Inconsistencia: embedding_dim declarado ({mode_v}) "
                f"no coincide con el ancho real de embedding_vector ({dims})."
            )


def run_semantic_search(params: SearchParams, out_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_dir = out_dir or _DEFAULT_OUTPUT_DIR
    ctx = SearchContext()
    validate_search_params(params)

    scored_path = _resolve_scored_path()
    scored = load_scored_dataset(scored_path)
    embeddings = load_embeddings_or_from_scored(scored)
    validate_input_schema(scored, embeddings)

    provider, model = _unique_provider_model(embeddings)

    merged = merge_scored_and_embeddings(scored, embeddings, ctx)
    total_available = len(merged)
    total_compared = total_available

    if total_available == 0:
        empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
        qprep = validate_and_prepare_query(params.query)
        report = {
            **_empty_report(params, provider, model, total_available, total_compared, 0, 0),
            "query_text": qprep,
            "filter_steps": [
                {"step": fl.step, "rows_in": fl.rows_in, "rows_out": fl.rows_out, "note": fl.note}
                for fl in ctx.filter_logs
            ],
        }
        files = _export_if_requested(params, out_dir, empty, report)
        print(
            "Sin empresas disponibles tras unir scored + embeddings "
            "(embedding_status=embedded y record_status válido).",
            file=sys.stderr,
        )
        _print_console_summary(
            {
                "query_text": qprep,
                "query_embedding_provider": provider,
                "query_embedding_model": model,
            },
            total_available,
            total_compared,
            0,
            0,
            files,
            None,
            filter_steps=report.get("filter_steps"),
        )
        return empty, report

    # Matriz de embeddings y similitud
    vectors = np.stack([_vector_to_1d(v) for v in merged["embedding_vector"].values], axis=0)
    _validate_embedding_matrix_vs_meta(merged, vectors)

    q_text = validate_and_prepare_query(params.query)
    q_meta = generate_query_embedding(q_text, provider, model)
    q_vec = q_meta["query_embedding"]
    if int(q_vec.shape[0]) != int(vectors.shape[1]):
        raise SearchAbort(
            f"Dimensión de embedding de consulta ({q_vec.shape[0]}) distinta a la de empresas ({vectors.shape[1]})."
        )

    merged = merged.copy()
    merged["semantic_similarity"] = cosine_similarity_matrix(vectors, q_vec)

    filtered = apply_filters(merged, params, ctx)
    rows_after_filters = len(filtered)

    if rows_after_filters == 0:
        report = {
            **_empty_report(
                params, provider, model, total_available, total_compared, rows_after_filters, 0,
                q_meta=q_meta,
            ),
            "filter_steps": [
                {"step": fl.step, "rows_in": fl.rows_in, "rows_out": fl.rows_out, "note": fl.note}
                for fl in ctx.filter_logs
            ],
        }
        print("No hay resultados tras aplicar los filtros; devolviendo estructura vacía.", file=sys.stderr)
        empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
        files = _export_if_requested(params, out_dir, empty, report)
        _print_console_summary(
            q_meta,
            total_available,
            total_compared,
            rows_after_filters,
            0,
            files,
            None,
            filter_steps=report.get("filter_steps"),
        )
        return empty, report

    ranked = compute_hybrid_ranking(filtered)
    explained = build_all_explanations(ranked)
    top_k = min(int(params.top_k), len(explained))
    top = explained.iloc[:top_k].copy()
    top["result_rank"] = np.arange(1, len(top) + 1, dtype=np.int64)

    final_df = format_final_results(top)

    avg_sim = float(pd.to_numeric(top["semantic_similarity"], errors="coerce").mean())
    avg_radar = float(pd.to_numeric(top["radar_score"], errors="coerce").mean())

    report = {
        "query_text": q_meta["query_text"],
        "total_companies_available": total_available,
        "total_companies_compared": total_compared,
        "total_results_after_filters": rows_after_filters,
        "top_k_returned": len(final_df),
        "embedding_provider": provider,
        "embedding_model": model,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "applied_filters": _applied_filters_dict(params),
        "ranking_formula_name": _RANKING_FORMULA_NAME,
        "average_similarity_of_returned_results": avg_sim,
        "average_radar_score_of_returned_results": avg_radar,
        "filter_steps": [
            {"step": fl.step, "rows_in": fl.rows_in, "rows_out": fl.rows_out, "note": fl.note}
            for fl in ctx.filter_logs
        ],
    }

    files = _export_if_requested(params, out_dir, final_df, report)

    _print_console_summary(
        q_meta,
        total_available,
        total_compared,
        rows_after_filters,
        len(final_df),
        files,
        final_df,
        filter_steps=report.get("filter_steps"),
    )

    return final_df, report


def _empty_report(
    params: SearchParams,
    provider: str,
    model: str,
    total_available: int,
    total_compared: int,
    after_filters: int,
    top_k_ret: int,
    q_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "query_text": (q_meta or {}).get("query_text", validate_and_prepare_query(params.query)),
        "total_companies_available": total_available,
        "total_companies_compared": total_compared,
        "total_results_after_filters": after_filters,
        "top_k_returned": top_k_ret,
        "embedding_provider": provider,
        "embedding_model": model,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "applied_filters": _applied_filters_dict(params),
        "ranking_formula_name": _RANKING_FORMULA_NAME,
        "average_similarity_of_returned_results": float("nan"),
        "average_radar_score_of_returned_results": float("nan"),
        "filter_steps": [],
    }


def _applied_filters_dict(params: SearchParams) -> dict[str, Any]:
    return {
        "top_k": params.top_k,
        "score_threshold": params.score_threshold,
        "similarity_threshold": params.similarity_threshold,
        "allowed_priority_labels": params.allowed_priority_labels,
        "allowed_sectors": params.allowed_sectors,
        "allowed_clusters": params.allowed_clusters,
        "exclude_discard": params.exclude_discard,
        "export_results": params.export_results,
    }


def _print_console_summary(
    q_meta: dict[str, Any],
    total_available: int,
    total_compared: int,
    after_filters: int,
    top_k_ret: int,
    files: Sequence[str],
    final_df: pd.DataFrame | None,
    filter_steps: list[dict[str, Any]] | None = None,
) -> None:
    print("--- Resumen búsqueda semántica ---")
    print(f"Consulta: {q_meta.get('query_text', '')!r}")
    print(f"Empresas disponibles (tras join y reglas base): {total_available}")
    print(f"Empresas comparadas (similitud coseno): {total_compared}")
    print(f"Empresas tras filtros opcionales: {after_filters}")
    if filter_steps:
        print("Detalle de filtros (filas antes -> después):")
        for fs in filter_steps:
            print(
                f"  - {fs.get('step', '')}: {fs.get('rows_in')} -> {fs.get('rows_out')}"
                + (f" ({fs.get('note')})" if fs.get("note") else "")
            )
    print(f"top_k devuelto: {top_k_ret}")
    print(f"Proveedor embedding: {q_meta.get('query_embedding_provider', '')}")
    print(f"Modelo embedding: {q_meta.get('query_embedding_model', '')}")
    if files:
        print("Archivos generados:")
        for f in files:
            print(f"  - {f}")
    if final_df is not None and len(final_df) > 0:
        print("\nMejores resultados:")
        for _, r in final_df.iterrows():
            brief = str(r.get("match_reason_summary", ""))[:120]
            print(
                f"  #{int(r['result_rank'])}  {r.get('company_name_display', '')}  "
                f"sim={float(r['semantic_similarity']):.3f}  "
                f"radar={float(pd.to_numeric(r['radar_score'], errors='coerce') or 0):.1f}  "
                f"prio={r.get('priority_label', '')}  "
                f"cluster={r.get('cluster_label', '')}  "
                f"razón: {brief}"
            )
    print("--- Fin resumen ---")


def _parse_list_arg(s: str | None) -> list[str] | None:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Búsqueda semántica sobre companies_scored + companies_embeddings.")
    p.add_argument("--query", "-q", required=True, help="Consulta en texto libre.")
    p.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help=f"Resultados a devolver (default {_DEFAULT_TOP_K}).")
    p.add_argument("--score-threshold", type=float, default=None, help="Mínimo radar_score (0-100).")
    p.add_argument("--similarity-threshold", type=float, default=None, help="Mínimo semantic_similarity (0-1).")
    p.add_argument(
        "--allowed-priority-labels",
        type=str,
        default=None,
        help="Comma-separated: CANDIDATE,REVIEW,DISCARD",
    )
    p.add_argument("--allowed-sectors", type=str, default=None, help="Subcadenas para sector_primary o priority_industry_labels, separadas por coma.")
    p.add_argument("--allowed-clusters", type=str, default=None, help="cluster_label permitidos, separados por coma.")
    p.add_argument("--exclude-discard", action="store_true", help="Excluir priority_label == DISCARD.")
    p.add_argument("--export-results", action="store_true", help="Exportar CSV, Parquet y JSON de reporte.")
    p.add_argument("--output-dir", type=str, default=None, help=f"Directorio de salida (default: {_DEFAULT_OUTPUT_DIR}).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR

    allowed_pl = _parse_list_arg(args.allowed_priority_labels)
    if allowed_pl:
        allowed_pl = [x.strip().upper() for x in allowed_pl]

    params = SearchParams(
        query=args.query,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        similarity_threshold=args.similarity_threshold,
        allowed_priority_labels=allowed_pl,
        allowed_sectors=_parse_list_arg(args.allowed_sectors),
        allowed_clusters=_parse_list_arg(args.allowed_clusters),
        exclude_discard=bool(args.exclude_discard),
        export_results=bool(args.export_results),
    )

    try:
        run_semantic_search(params, out_dir=out_dir)
    except SearchAbort as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
