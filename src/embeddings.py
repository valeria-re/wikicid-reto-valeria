"""
Generación de embeddings desde companies_clean.csv (pipeline de producción).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
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
    "duplicate_action",
)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")

PLACEHOLDERS = frozenset(
    {
        "n/d",
        "nd",
        "n/a",
        "na",
        "null",
        "none",
        "",
    }
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    s = v.strip()
    return s if s else default


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    batch_size: int
    max_text_length: int
    min_text_length: int
    max_retries: int
    input_path: Path
    output_dir: Path
    openai_api_key: str | None
    incremental: bool

    @classmethod
    def from_env(cls, explicit: dict[str, Any] | None = None) -> EmbeddingConfig:
        ex = explicit or {}
        root = Path(ex.get("project_root") or _env_str("PROJECT_ROOT") or str(_REPO_ROOT))
        provider = (ex.get("embedding_provider") or _env_str("EMBEDDING_PROVIDER", "local") or "local").lower()
        model = ex.get("embedding_model") or _env_str("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
        batch_size = int(ex.get("embedding_batch_size") or _env_int("EMBEDDING_BATCH_SIZE", 32))
        max_text_length = int(ex.get("embedding_max_text_length") or _env_int("EMBEDDING_MAX_TEXT_LENGTH", 8000))
        min_text_length = int(ex.get("embedding_min_text_length") or _env_int("EMBEDDING_MIN_TEXT_LENGTH", 32))
        max_retries = int(ex.get("embedding_max_retries") or _env_int("EMBEDDING_MAX_RETRIES", 3))
        default_in = root / "outputs" / "companies_clean.csv"
        input_path = Path(ex.get("input_path") or _env_str("COMPANIES_CLEAN_CSV") or str(default_in))
        if not input_path.is_absolute():
            input_path = (root / input_path).resolve()
        output_dir = Path(ex.get("output_dir") or _env_str("EMBEDDING_OUTPUT_DIR", "outputs"))
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        openai_api_key = ex.get("openai_api_key") or _env_str("OPENAI_API_KEY")
        incremental = str(ex.get("incremental") or _env_str("EMBEDDING_INCREMENTAL", "1") or "1").lower() in (
            "1",
            "true",
            "yes",
        )
        return cls(
            provider=provider,
            model=model,
            batch_size=max(1, batch_size),
            max_text_length=max(1, max_text_length),
            min_text_length=max(1, min_text_length),
            max_retries=max(1, max_retries),
            input_path=input_path,
            output_dir=output_dir,
            openai_api_key=openai_api_key,
            incremental=incremental,
        )


def load_clean_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path.resolve()}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.replace("", np.nan)
    logger.info("Filas leídas: %s", len(df))
    return df


def validate_input_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el dataset: {missing}. "
            f"Columnas presentes: {list(df.columns)}"
        )


def _is_blank_company_id(v: Any) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    return s == ""


def _semantic_text_raw_nonempty(v: Any) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip() != ""


def filter_eligible_rows(df: pd.DataFrame, cfg: EmbeddingConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    skipped_records: list[dict[str, Any]] = []
    mask = pd.Series(True, index=df.index)

    for idx, row in df.iterrows():
        reason: str | None = None
        if str(row.get("duplicate_action", "")).strip() == "dropped_duplicate":
            reason = "duplicate_action=dropped_duplicate"
        elif _is_blank_company_id(row.get("company_id")):
            reason = "company_id vacío"
        elif not _semantic_text_raw_nonempty(row.get("semantic_text")):
            reason = "semantic_text vacío"
        else:
            rs = str(row.get("record_status", "")).strip()
            if rs == "invalid_record":
                reason = "record_status=invalid_record"
            elif rs == "valid_record":
                pass
            elif rs == "partial_record":
                raw = str(row.get("semantic_text", "")).strip()
                if len(raw) < cfg.min_text_length:
                    reason = "partial_record sin semantic_text suficiente"
            else:
                reason = f"record_status no soportado: {rs}"

        if reason:
            mask.loc[idx] = False
            skipped_records.append(_failed_row_dict(row, reason, stage="eligibility_filter"))

    eligible = df.loc[mask].copy()
    logger.info("Filas elegibles (filtro inicial): %s", len(eligible))
    return eligible, skipped_records


def _failed_row_dict(
    row: pd.Series,
    reason: str,
    *,
    stage: str,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    return {
        "company_id": row.get("company_id", ""),
        "company_name_display": row.get("company_name_display", ""),
        "semantic_text": row.get("semantic_text", ""),
        "reason": reason,
        "provider": provider,
        "model": model,
        "stage": stage,
    }


def strip_urls(text: str) -> str:
    return _URL_PATTERN.sub("", text)


def normalize_text_for_validation(text: str) -> str:
    t = text.strip()
    t = _MULTI_SPACE.sub(" ", t)
    return t.strip()


def is_placeholder_text(text: str) -> bool:
    compact = normalize_text_for_validation(text).lower()
    return compact in PLACEHOLDERS


def prepare_text_for_embedding(raw: Any, cfg: EmbeddingConfig) -> tuple[str | None, str | None]:
    """
    Prepara texto desde semantic_text (sin reconstruir desde otras columnas).
    Retorna (texto_ok, razón_fallo).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, "semantic_text nulo"
    s = str(raw).rstrip()
    if not s.strip():
        return None, "semantic_text vacío tras limpieza inicial"
    s = normalize_text_for_validation(s)
    s = strip_urls(s)
    s = normalize_text_for_validation(s)
    if not s:
        return None, "texto vacío tras quitar URLs"
    if is_placeholder_text(s):
        return None, "texto tipo placeholder (N/D, N/A, etc.)"
    if len(s) < cfg.min_text_length:
        return None, f"texto demasiado corto (< {cfg.min_text_length} caracteres)"
    return s, None


def truncate_text_utf8_safe(text: str, max_chars: int) -> tuple[str, int, int, bool]:
    if len(text) <= max_chars:
        return text, len(text), len(text), False
    # Índices por caracteres Unicode (no cortar surrogates)
    cut = text[:max_chars]
    # Evitar cortar en medio de secuencia si hubiera rareza: en Python 3 str slice es por codepoints
    return cut, len(text), len(cut), True


class EmbeddingBackend:
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalSentenceBackend(EmbeddingBackend):
    def __init__(self, model_name: str, encode_batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._encode_batch_size = max(1, encode_batch_size)
        self._dim: int | None = None

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=min(len(texts), self._encode_batch_size),
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if self._dim is None:
            self._dim = int(arr.shape[1])
        return arr

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimensión desconocida antes del primer batch")
        return self._dim


class OpenAIBackend(EmbeddingBackend):
    def __init__(self, model: str, api_key: str) -> None:
        import httpx

        if not api_key:
            raise ValueError("OPENAI_API_KEY es obligatoria para EMBEDDING_PROVIDER=openai")
        self._model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=120.0)
        self._dim: int | None = None

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        r = self._client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
        data = sorted(data, key=lambda x: x["index"])
        vectors = [np.asarray(d["embedding"], dtype=np.float32) for d in data]
        out = np.vstack(vectors)
        if self._dim is None:
            self._dim = int(out.shape[1])
        return out

    def close(self) -> None:
        self._client.close()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimensión desconocida antes del primer batch")
        return self._dim


def _embedding_dimension_known(backend: EmbeddingBackend) -> int | None:
    try:
        return int(backend.dimension)
    except Exception:
        return None


def get_embedding_backend(cfg: EmbeddingConfig) -> EmbeddingBackend:
    p = cfg.provider.lower()
    if p == "local":
        return LocalSentenceBackend(cfg.model, encode_batch_size=cfg.batch_size)
    if p == "openai":
        key = cfg.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY no configurada (requerida para proveedor openai).")
        return OpenAIBackend(cfg.model, key)
    raise ValueError(
        f"EMBEDDING_PROVIDER inválido: {cfg.provider!r}. Use 'openai' o 'local'."
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_embedding_cache(
    parquet_path: Path, provider: str, model: str
) -> dict[tuple[str, str, str, str], np.ndarray]:
    if not parquet_path.is_file():
        return {}
    try:
        prev = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning("No se pudo leer caché incremental %s: %s", parquet_path, e)
        return {}
    need = {"company_id", "semantic_text_hash", "embedding_provider", "embedding_model", "embedding_vector"}
    if not need.issubset(set(prev.columns)):
        return {}
    cache: dict[tuple[str, str, str, str], np.ndarray] = {}
    for _, r in prev.iterrows():
        if str(r.get("embedding_provider")) != provider or str(r.get("embedding_model")) != model:
            continue
        cid = str(r["company_id"])
        th = str(r["semantic_text_hash"])
        key = (cid, th, provider, model)
        ev = r["embedding_vector"]
        if isinstance(ev, np.ndarray):
            vec = ev.astype(np.float32, copy=False)
        else:
            vec = np.asarray(ev, dtype=np.float32)
        cache[key] = vec
    return cache


def embed_batch_with_retries(
    backend: EmbeddingBackend,
    texts: list[str],
    cfg: EmbeddingConfig,
) -> tuple[np.ndarray | None, str | None]:
    last_err: str | None = None
    for attempt in range(cfg.max_retries):
        try:
            return backend.embed_batch(texts), None
        except Exception as e:
            last_err = str(e)
            logger.warning(
                "Lote falló (intento %s/%s): %s",
                attempt + 1,
                cfg.max_retries,
                e,
            )
            if attempt < cfg.max_retries - 1:
                time.sleep(2**attempt)
    return None, last_err or "error desconocido en embedding"


def embed_single_with_retries(
    backend: EmbeddingBackend,
    text: str,
    cfg: EmbeddingConfig,
) -> tuple[np.ndarray | None, str | None]:
    return embed_batch_with_retries(backend, [text], cfg)


def _chunks(indices: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def save_embeddings_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_vector_index_structure(
    company_ids: list[str],
    matrix: np.ndarray,
    meta: dict[str, Any],
    index_dir: Path,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embedding_matrix.npy", matrix.astype(np.float32))
    (index_dir / "company_id_order.json").write_text(
        json.dumps(company_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (index_dir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_failures_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(
            columns=[
                "company_id",
                "company_name_display",
                "semantic_text",
                "reason",
                "provider",
                "model",
                "stage",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def export_manifest(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(cfg: EmbeddingConfig | None = None) -> None:
    cfg = cfg or EmbeddingConfig.from_env()
    ts = datetime.now(timezone.utc).isoformat()

    df_in = load_clean_dataset(cfg.input_path)
    validate_input_schema(df_in)
    total_rows_input = len(df_in)

    eligible_df, eligibility_fails = filter_eligible_rows(df_in, cfg)
    total_rows_eligible = len(eligible_df)

    backend = get_embedding_backend(cfg)
    try:
        _run_embedding_core(
            cfg,
            eligible_df,
            total_rows_input,
            total_rows_eligible,
            eligibility_fails,
            ts,
            backend,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _run_embedding_core(
    cfg: EmbeddingConfig,
    eligible_df: pd.DataFrame,
    total_rows_input: int,
    total_rows_eligible: int,
    eligibility_fails: list[dict[str, Any]],
    ts: str,
    backend: EmbeddingBackend,
) -> None:
    cache: dict[tuple[str, str, str, str], np.ndarray] = {}
    parquet_out = cfg.output_dir / "companies_embeddings.parquet"
    if cfg.incremental:
        cache = load_existing_embedding_cache(parquet_out, cfg.provider, cfg.model)
        if cache:
            logger.info("Caché incremental: %s entradas reutilizables", len(cache))

    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = list(eligibility_fails)

    work_items: list[dict[str, Any]] = []
    for idx, row in eligible_df.iterrows():
        prepared, reason = prepare_text_for_embedding(row.get("semantic_text"), cfg)
        if prepared is None:
            failures.append(
                _failed_row_dict(
                    row,
                    reason or "validación de texto",
                    stage="text_validation",
                    provider=cfg.provider,
                    model=cfg.model,
                )
            )
            continue
        truncated, orig_len, final_len, was_trunc = truncate_text_utf8_safe(prepared, cfg.max_text_length)
        th = text_hash(truncated)
        work_items.append(
            {
                "row": row,
                "semantic_text_used": truncated,
                "semantic_text_hash": th,
                "text_original_length": orig_len,
                "text_final_length": final_len,
                "text_was_truncated": was_trunc,
            }
        )

    embedded_count = 0
    skipped_cache_count = 0

    to_compute: list[dict[str, Any]] = []
    for item in work_items:
        row = item["row"]
        cid = str(row["company_id"]).strip()
        key = (cid, item["semantic_text_hash"], cfg.provider, cfg.model)
        if cfg.incremental and key in cache:
            vec = cache[key]
            dim = int(vec.shape[0])
            rows_out.append(
                _build_output_row(
                    row,
                    item,
                    vec,
                    dim,
                    cfg,
                    embedding_status="skipped",
                    source_record_status=str(row.get("record_status", "")),
                )
            )
            skipped_cache_count += 1
        else:
            to_compute.append(item)

    batch_indices = list(range(len(to_compute)))
    for batch_ixs in _chunks(batch_indices, cfg.batch_size):
        batch = [to_compute[i] for i in batch_ixs]
        texts = [b["semantic_text_used"] for b in batch]
        arr, err = embed_batch_with_retries(backend, texts, cfg)
        if arr is None:
            logger.warning(
                "Lote de %s textos falló tras reintentos; degradando a procesamiento individual",
                len(batch),
            )
            for item in batch:
                one, err_one = embed_single_with_retries(backend, item["semantic_text_used"], cfg)
                row = item["row"]
                if one is None:
                    failures.append(
                        _failed_row_dict(
                            row,
                            err_one or err or "error de embedding",
                            stage="embedding",
                            provider=cfg.provider,
                            model=cfg.model,
                        )
                    )
                    continue
                vec = one[0]
                dim = int(vec.shape[0])
                rows_out.append(
                    _build_output_row(
                        row,
                        item,
                        vec,
                        dim,
                        cfg,
                        embedding_status="embedded",
                        source_record_status=str(row.get("record_status", "")),
                    )
                )
                embedded_count += 1
            continue

        dim = int(arr.shape[1])
        for k, item in enumerate(batch):
            row = item["row"]
            vec = arr[k]
            rows_out.append(
                _build_output_row(
                    row,
                    item,
                    vec,
                    dim,
                    cfg,
                    embedding_status="embedded",
                    source_record_status=str(row.get("record_status", "")),
                )
            )
            embedded_count += 1

    if rows_out:
        out_df = pd.DataFrame(rows_out)
        index_ids: list[str] = []
        index_vecs: list[np.ndarray] = []
        for r in rows_out:
            index_ids.append(str(r["company_id"]))
            ev = r["embedding_vector"]
            index_vecs.append(np.asarray(ev, dtype=np.float32))
        matrix = np.vstack(index_vecs)
        build_vector_index_structure(
            index_ids,
            matrix,
            {
                "embedding_provider": cfg.provider,
                "embedding_model": cfg.model,
                "embedding_dimension": int(matrix.shape[1]),
                "n_vectors": int(matrix.shape[0]),
                "order_note": "Orden de filas = orden en company_id_order.json",
            },
            cfg.output_dir / "companies_vector_index",
        )
        save_embeddings_parquet(out_df, parquet_out)
    else:
        export_failures_csv(failures, cfg.output_dir / "companies_embeddings_failed.csv")
        export_manifest(
            {
                "input_file": str(cfg.input_path.resolve()),
                "total_rows_input": total_rows_input,
                "total_rows_eligible": total_rows_eligible,
                "total_rows_embedded": 0,
                "total_rows_skipped": 0,
                "total_rows_failed": len(failures),
                "embedding_provider": cfg.provider,
                "embedding_model": cfg.model,
                "embedding_dimension": _embedding_dimension_known(backend),
                "batch_size": cfg.batch_size,
                "max_text_length": cfg.max_text_length,
                "execution_timestamp": ts,
            },
            cfg.output_dir / "embedding_manifest.json",
        )
        logger.info(
            "Resumen: leídas=%s elegibles=%s embebidas=0 saltadas(cache)=%s fallidas=%s proveedor=%s modelo=%s dim=%s archivos=%s",
            total_rows_input,
            total_rows_eligible,
            skipped_cache_count,
            len(failures),
            cfg.provider,
            cfg.model,
            "n/a",
            f"{cfg.output_dir / 'companies_embeddings_failed.csv'}, {cfg.output_dir / 'embedding_manifest.json'}",
        )
        return

    total_failed = len(failures)
    dim_final = int(np.asarray(rows_out[0]["embedding_vector"]).shape[-1]) if rows_out else None

    export_failures_csv(failures, cfg.output_dir / "companies_embeddings_failed.csv")
    export_manifest(
        {
            "input_file": str(cfg.input_path.resolve()),
            "total_rows_input": total_rows_input,
            "total_rows_eligible": total_rows_eligible,
            "total_rows_embedded": embedded_count,
            "total_rows_skipped": skipped_cache_count,
            "total_rows_failed": total_failed,
            "embedding_provider": cfg.provider,
            "embedding_model": cfg.model,
            "embedding_dimension": dim_final,
            "batch_size": cfg.batch_size,
            "max_text_length": cfg.max_text_length,
            "execution_timestamp": ts,
        },
        cfg.output_dir / "embedding_manifest.json",
    )

    logger.info(
        "Resumen: leídas=%s elegibles=%s embebidas=%s saltadas(cache)=%s fallidas=%s proveedor=%s modelo=%s dim=%s archivos=%s",
        total_rows_input,
        total_rows_eligible,
        embedded_count,
        skipped_cache_count,
        total_failed,
        cfg.provider,
        cfg.model,
        dim_final,
        f"{parquet_out}, {cfg.output_dir / 'companies_embeddings_failed.csv'}, "
        f"{cfg.output_dir / 'embedding_manifest.json'}, {cfg.output_dir / 'companies_vector_index'}",
    )


def _build_output_row(
    row: pd.Series,
    item: dict[str, Any],
    vec: np.ndarray,
    dim: int,
    cfg: EmbeddingConfig,
    *,
    embedding_status: str,
    source_record_status: str,
) -> dict[str, Any]:
    vlist = vec.flatten().tolist()
    return {
        "company_id": row.get("company_id"),
        "company_name_display": row.get("company_name_display"),
        "company_name_normalized": row.get("company_name_normalized"),
        "sector_primary": row.get("sector_primary"),
        "website_domain": row.get("website_domain"),
        "semantic_text_used": item["semantic_text_used"],
        "semantic_text_hash": item["semantic_text_hash"],
        "embedding_vector": vlist,
        "embedding_dim": dim,
        "embedding_provider": cfg.provider,
        "embedding_model": cfg.model,
        "embedding_status": embedding_status,
        "text_original_length": item["text_original_length"],
        "text_final_length": item["text_final_length"],
        "text_was_truncated": item["text_was_truncated"],
        "source_record_status": source_record_status,
    }


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
