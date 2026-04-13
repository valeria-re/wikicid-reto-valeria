import os
import time
from pathlib import Path

import cohere
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_CSV = OUTPUT_DIR / "cleaned_companies.csv"
EMB_NPY = OUTPUT_DIR / "company_embeddings.npy"

BATCH_SIZE = 20              # 20 registros por request
REQUESTS_PER_MINUTE = 10     # 10 requests por minuto
SLEEP_SECONDS = 60 / REQUESTS_PER_MINUTE  # 6 segundos entre requests


def load_cleaned_data() -> pd.DataFrame:
    if not CLEAN_CSV.is_file():
        raise FileNotFoundError(f"No se encontró {CLEAN_CSV}")
    return pd.read_csv(CLEAN_CSV, encoding="utf-8")


def load_client() -> cohere.ClientV2:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable de entorno COHERE_API_KEY")
    return cohere.ClientV2(api_key=api_key)


def generate_embeddings(texts: list[str], client: cohere.ClientV2) -> np.ndarray:
    vectors = []
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        response = client.embed(
            model="embed-v4.0",
            input_type="search_document",
            texts=batch,
            embedding_types=["float"],
        )

        batch_vecs = np.asarray(response.embeddings.float, dtype=np.float32)
        vectors.append(batch_vecs)

        processed = min(i + BATCH_SIZE, total)
        print(f"Procesados {processed}/{total}")

        if processed < total:
            time.sleep(SLEEP_SECONDS)

    return np.vstack(vectors)


def save_embeddings(arr: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_NPY, arr)


def main() -> None:
    df = load_cleaned_data()
    texts = df["search_text"].fillna("").astype(str).tolist()

    client = load_client()
    emb = generate_embeddings(texts, client)
    save_embeddings(emb)

    print(f"Rows: {len(df)}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Saved to: {EMB_NPY}")


if __name__ == "__main__":
    main()