from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"
_SCORED_CSV = _DEFAULT_OUTPUT_DIR / "scored_companies.csv"
_EMBEDDINGS_NPY = _DEFAULT_OUTPUT_DIR / "company_embeddings.npy"

REQUIRED_SCORED_COLS: tuple[str, ...] = (
    "name",
    "description",
    "website",
    "sector",
    "search_text",
    "radar_score",
    "radar_label",
    "score_reason",
)

VALID_PRIORITY_LABELS = frozenset({"CANDIDATE", "REVIEW", "DISCARD"})
_DEFAULT_TOP_K = 20


class SearchAbort(Exception):
    """Error de validación o datos."""


@dataclass
class SearchParams:
    query: str
    top_k: int = _DEFAULT_TOP_K
    score_threshold: float | None = None
    similarity_threshold: float | None = None
    allowed_priority_labels: list[str] | None = None
    allowed_sectors: list[str] | None = None
    exclude_discard: bool = False
    export_results: bool = False


def _resolve_scored_path() -> Path:
    if _SCORED_CSV.is_file():
        return _SCORED_CSV
    raise SearchAbort(f"Falta el archivo `{_SCORED_CSV}`.")


def load_scored_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SearchAbort(f"No se encontró el dataset scored: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def load_embeddings(path: Path = _EMBEDDINGS_NPY) -> np.ndarray:
    if not path.is_file():
        raise SearchAbort(f"No se encontró el archivo de embeddings: {path}")
    try:
        arr = np.load(path)
    except Exception as e:
        raise SearchAbort(f"No se pudieron cargar los embeddings: {e}")
    if arr.ndim != 2:
        raise SearchAbort(f"Los embeddings deben ser 2D. Shape recibido: {arr.shape}")
    return arr.astype(np.float64)


def validate_input_schema(scored: pd.DataFrame, embeddings: np.ndarray) -> None:
    missing = [c for c in REQUIRED_SCORED_COLS if c not in scored.columns]
    if missing:
        raise SearchAbort(
            f"Faltan columnas obligatorias en el dataset puntuado: {missing}. "
            f"Columnas presentes: {list(scored.columns)}"
        )
    if len(scored) != embeddings.shape[0]:
        raise SearchAbort(
            f"No coincide el número de filas entre scored ({len(scored)}) y embeddings ({embeddings.shape[0]})."
        )


def validate_search_params(params: SearchParams) -> None:
    if not params.query or not str(params.query).strip():
        raise SearchAbort("La consulta no puede estar vacía.")
    if not isinstance(params.top_k, int) or params.top_k < 1:
        raise SearchAbort("`top_k` debe ser un entero positivo.")
    if params.score_threshold is not None and not (0 <= float(params.score_threshold) <= 100):
        raise SearchAbort("`score_threshold` debe estar entre 0 y 100.")
    if params.similarity_threshold is not None and not (0 <= float(params.similarity_threshold) <= 1):
        raise SearchAbort("`similarity_threshold` debe estar entre 0 y 1.")
    if params.allowed_priority_labels:
        bad = [x for x in params.allowed_priority_labels if x not in VALID_PRIORITY_LABELS]
        if bad:
            raise SearchAbort(f"Etiquetas inválidas en allowed_priority_labels: {bad}")


def get_query_embedding(query_text: str) -> np.ndarray:
    import os
    import cohere
    import numpy as np

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise SearchAbort("Falta la variable de entorno COHERE_API_KEY.")

    try:
        client = cohere.ClientV2(api_key=api_key)

        response = client.embed(
            model="embed-v4.0",
            input_type="search_query",
            texts=[query_text],
            embedding_types=["float"],
        )

        vec = np.asarray(response.embeddings.float[0], dtype=np.float64).reshape(-1)
        if vec.size == 0:
            raise SearchAbort("Cohere devolvió un embedding vacío para la consulta.")

        return vec

    except Exception as e:
        raise SearchAbort(f"No se pudo generar el embedding con Cohere: {e}")

def cosine_similarity_matrix(company_matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    x = np.asarray(company_matrix, dtype=np.float64)
    q = np.asarray(query_vec, dtype=np.float64).reshape(-1)

    if x.ndim != 2:
        raise SearchAbort(f"La matriz de embeddings debe ser 2D. Shape: {x.shape}")
    if x.shape[1] != q.shape[0]:
        raise SearchAbort(
            f"Dimensión incompatible: empresas d={x.shape[1]}, consulta d={q.shape[0]}"
        )

    row_norms = np.linalg.norm(x, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-12)
    x_unit = x / row_norms

    q_norm = np.linalg.norm(q)
    q_unit = q / max(q_norm, 1e-12)

    cos_raw = (x_unit @ q_unit).ravel()
    semantic_01 = (cos_raw + 1.0) / 2.0
    return np.clip(semantic_01, 0.0, 1.0)


def apply_filters(df: pd.DataFrame, params: SearchParams) -> pd.DataFrame:
    out = df.copy()

    if params.score_threshold is not None:
        out = out.loc[pd.to_numeric(out["radar_score"], errors="coerce").fillna(-np.inf) >= float(params.score_threshold)]

    if params.similarity_threshold is not None:
        out = out.loc[pd.to_numeric(out["semantic_similarity"], errors="coerce").fillna(0) >= float(params.similarity_threshold)]

    if params.allowed_priority_labels:
        allowed = set(params.allowed_priority_labels)
        out = out.loc[out["radar_label"].astype(str).isin(allowed)]

    if params.exclude_discard:
        out = out.loc[out["radar_label"].astype(str).str.strip() != "DISCARD"]

    if params.allowed_sectors:
        sectors = [s.strip().lower() for s in params.allowed_sectors if str(s).strip()]
        out = out.loc[out["sector"].astype(str).str.lower().apply(lambda x: any(s in x for s in sectors))]

    return out.copy()


def compute_hybrid_ranking(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sem = pd.to_numeric(out["semantic_similarity"], errors="coerce").fillna(0.0).to_numpy()
    radar = pd.to_numeric(out["radar_score"], errors="coerce").fillna(0.0).to_numpy()
    radar_n = np.clip(radar / 100.0, 0.0, 1.0)

    priority_map = {"CANDIDATE": 1.0, "REVIEW": 0.5, "DISCARD": 0.0}
    prio = out["radar_label"].astype(str).str.strip().map(lambda x: priority_map.get(x, 0.0)).to_numpy()

    ranking_score = (0.75 * sem) + (0.20 * radar_n * sem) + (0.05 * prio * sem)

    out["ranking_score"] = ranking_score
    out = out.sort_values(["ranking_score", "semantic_similarity"], ascending=False, kind="mergesort").reset_index(drop=True)
    out["result_rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def build_all_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["semantic_match_reason"] = out.apply(
        lambda r: f"Alineación semántica con la consulta dentro del sector '{r.get('sector', '')}'.",
        axis=1,
    )
    out["strategic_match_reason"] = out.apply(
        lambda r: f"Clasificación estratégica {r.get('radar_label', '')} con radar {r.get('radar_score', '')}.",
        axis=1,
    )
    out["match_reason_summary"] = out.apply(
        lambda r: f"Empresa relevante por similitud semántica y contexto de negocio en '{r.get('sector', '')}'.",
        axis=1,
    )
    return out


RESULT_COLUMNS: tuple[str, ...] = (
    "result_rank",
    "name",
    "semantic_similarity",
    "ranking_score",
    "radar_score",
    "radar_label",
    "sector",
    "website",
    "description",
    "score_reason",
    "semantic_match_reason",
    "strategic_match_reason",
    "match_reason_summary",
)


def format_final_results(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    return df[cols].copy()


def export_results(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "semantic_search_results.csv"
    json_p = out_dir / "semantic_search_results_preview.json"

    df.to_csv(csv_p, index=False, encoding="utf-8")
    json_p.write_text(
        df.head(20).to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_p, json_p


def generate_report_json(payload: dict[str, Any], out_dir: Path) -> Path:
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
    csv_p, preview_p = export_results(final_df, out_dir)
    js_p = generate_report_json(report, out_dir)
    return [str(csv_p), str(preview_p), str(js_p)]


def run_semantic_search(params: SearchParams, out_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_dir = out_dir or _DEFAULT_OUTPUT_DIR
    validate_search_params(params)

    scored_path = _resolve_scored_path()
    scored = load_scored_dataset(scored_path)
    embeddings = load_embeddings(_EMBEDDINGS_NPY)
    validate_input_schema(scored, embeddings)

    query_text = str(params.query).strip()
    query_vec = get_query_embedding(query_text)

    if query_vec.shape[0] != embeddings.shape[1]:
        raise SearchAbort(
            f"Dimensión incompatible entre consulta ({query_vec.shape[0]}) y embeddings ({embeddings.shape[1]})."
        )

    df = scored.copy()
    df["semantic_similarity"] = cosine_similarity_matrix(embeddings, query_vec)

    filtered = apply_filters(df, params)
    ranked = compute_hybrid_ranking(filtered)
    explained = build_all_explanations(ranked)

    top_k = min(int(params.top_k), len(explained))
    final_df = format_final_results(explained.iloc[:top_k].copy())

    report = {
        "query_text": query_text,
        "total_companies_available": len(df),
        "total_results_after_filters": len(filtered),
        "top_k_returned": len(final_df),
        "embedding_provider": "sentence-transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "average_similarity_of_returned_results": float(pd.to_numeric(final_df["semantic_similarity"], errors="coerce").mean()) if len(final_df) else float("nan"),
        "average_radar_score_of_returned_results": float(pd.to_numeric(final_df["radar_score"], errors="coerce").mean()) if len(final_df) else float("nan"),
    }

    _export_if_requested(params, out_dir, final_df, report)
    return final_df, report


def _parse_list_arg(s: str | None) -> list[str] | None:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Búsqueda semántica sobre scored_companies.csv + company_embeddings.npy.")
    p.add_argument("--query", "-q", required=True, help="Consulta en texto libre.")
    p.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help=f"Resultados a devolver (default {_DEFAULT_TOP_K}).")
    p.add_argument("--score-threshold", type=float, default=None, help="Mínimo radar_score (0-100).")
    p.add_argument("--similarity-threshold", type=float, default=None, help="Mínimo semantic_similarity (0-1).")
    p.add_argument("--allowed-priority-labels", type=str, default=None, help="Comma-separated: CANDIDATE,REVIEW,DISCARD")
    p.add_argument("--allowed-sectors", type=str, default=None, help="Sectores permitidos separados por coma.")
    p.add_argument("--exclude-discard", action="store_true", help="Excluir radar_label == DISCARD.")
    p.add_argument("--export-results", action="store_true", help="Exportar CSV y JSON de reporte.")
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
        exclude_discard=bool(args.exclude_discard),
        export_results=bool(args.export_results),
    )

    try:
        final_df, report = run_semantic_search(params, out_dir=out_dir)
        print("--- Resumen búsqueda semántica ---")
        print(f"Consulta: {report['query_text']!r}")
        print(f"Empresas disponibles: {report['total_companies_available']}")
        print(f"Resultados tras filtros: {report['total_results_after_filters']}")
        print(f"Top devuelto: {report['top_k_returned']}")
        if len(final_df) > 0:
            print("\nMejores resultados:")
            for _, r in final_df.iterrows():
                print(
                    f"  #{int(r['result_rank'])} {r.get('name', '')} "
                    f"sim={float(r['semantic_similarity']):.3f} "
                    f"radar={float(pd.to_numeric(r['radar_score'], errors='coerce') or 0):.1f} "
                    f"label={r.get('radar_label', '')}"
                )
        print("--- Fin resumen ---")
    except SearchAbort as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())