"""
Aplicación web: búsqueda semántica vía `src/search.py` (mismo pipeline que CLI).
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search import SearchAbort, SearchParams, run_semantic_search

app = Flask(__name__)


def _domain_to_website(url_or_domain: str) -> str:
    s = (url_or_domain or "").strip()
    if not s:
        return ""
    if s.startswith(("http://", "https://")):
        return s
    return f"https://{s}"


def dataframe_rows_for_template(df):
    """Convierte el DataFrame de `run_semantic_search` al dict que espera `index.html`."""
    rows = []
    for _, row in df.iterrows():
        website = str(row.get("website") or "").strip()
        desc = str(row.get("description") or "").strip()

        rows.append(
            {
                "name": str(row.get("name") or ""),
                "sector": str(row.get("sector") or "Sector desconocido"),
                "website": _domain_to_website(website),
                "radar_score": round(float(row.get("radar_score") or 0), 2),
                "radar_label": str(row.get("radar_label") or ""),
                "description": desc,
                "score_reason": str(row.get("score_reason") or ""),
                "similarity": round(float(row.get("semantic_similarity") or 0), 3),
            }
        )
    return rows


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=None, query=None, error=None)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    results = None
    error = None

    if query and str(query).strip():
        try:
            params = SearchParams(query=str(query).strip(), top_k=10, export_results=False)
            final_df, _report = run_semantic_search(params, out_dir=BASE_DIR / "outputs")
            results = sorted(dataframe_rows_for_template(final_df),key=lambda x: x["similarity"],reverse=True)
        except SearchAbort as e:
            error = str(e)
        except Exception as e:
            error = f"Error en la búsqueda: {e}"

    return render_template("index.html", results=results, query=query, error=error)


if __name__ == "__main__":
    app.run(debug=True)
