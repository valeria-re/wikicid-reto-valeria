# Mini Radar + Semantic Company Search

## Overview

Mini Radar + Semantic Company Search is an AI-assisted company intelligence prototype designed to help users discover relevant businesses through natural language queries.

The system combines:

- Semantic Search (embeddings + cosine similarity)
- Strategic Radar Scoring
- Industry Prioritization
- Business Impact Detection
- Data Quality Evaluation
- Premium Flask Web Interface

It allows users to search companies using plain language queries such as:

> "Platforms that improve patient experience"

And returns ranked companies based on:

1. Semantic relevance  
2. Strategic fit  
3. Radar score quality  

---

# Live Demo

Current deployed version (Render):

https://wikicid-reto-valeria.onrender.com

⸻

Important Hosting Note

This project is currently hosted on Render.

During deployment, the original local embedding model (sentence-transformers) consumed too much RAM for the free Render plan. Because of that limitation, the embedding system was upgraded to use the Cohere Embeddings API, which significantly reduced memory usage and allowed stable deployment.

Therefore:

Anyone who wants to run or deploy this project must create their own API key from:

https://dashboard.cohere.com/

And configure it in their environment variables.

Example:

COHERE_API_KEY=your_api_key_here


⸻

Main Features

Semantic Search

Users can type natural language queries and retrieve the most relevant companies using embeddings.

Examples:
	•	fintechs for digital payments
	•	healthcare platforms for hospitals
	•	telecom customer experience solutions
	•	AI tools for education

⸻

Strategic Radar Score (0–100)

Each company receives a score based on:

1. Industry Alignment (35 pts)

Measures whether the company operates in priority sectors such as:
	•	Telecom
	•	Banking
	•	Fintech
	•	Healthcare
	•	Education
	•	Media
	•	Agriculture
	•	Mining
	•	Energy

⸻

2. Business Impact Alignment (35 pts)

Detects signals related to:
	•	Revenue growth
	•	Customer experience
	•	Operational efficiency
	•	Automation
	•	Cost reduction

⸻

3. Data Maturity (30 pts)

Measures data quality using:
	•	Website validity
	•	Description richness
	•	Sector completeness
	•	Search text completeness

⸻

Priority Labels

Score	Label
> 80	CANDIDATE
40–80	REVIEW
< 40	DISCARD


⸻
```text
Project Architecture

mini-radar/
│
├── app.py
├── requirements.txt
├── README.md
├── AI_USAGE.md
│
├── data/
│   └── companies.xlsx
│
├── outputs/
│   ├── ...
│
├── src/
│   ├── cleaning.py
│   ├── embeddings.py
│   ├── scoring_clustering.py
│   └── semantic_search.py
│
├── templates/
│   └── index.html
│
└── static/
    └── styles.css

```
⸻

Pipeline Flow

Step 1 — Data Cleaning

Reads the raw Excel dataset and transforms it into a normalized structured dataset.

python src/cleaning.py

Output:

outputs/companies_clean.csv


⸻

Step 2 — Embeddings Generation (Cohere API)

Creates semantic vectors using Cohere API.

python src/embeddings.py

Output:

outputs/companies_embeddings.parquet


⸻

Step 3 — Radar Scoring + Clustering

Scores companies strategically and groups them into thematic clusters.

python src/scoring.py

Output:

outputs/companies_scored.csv


⸻

Step 4 — Semantic Search

python src/search.py --query "healthcare platforms"


⸻

Step 5 — Flask App

python app.py


⸻

Environment Variables Required

Create a .env file or configure hosting variables:

COHERE_API_KEY=your_api_key_here
FLASK_ENV=production


⸻

Installation

Clone Repo

git clone <your-repo-url>
cd mini-radar


⸻

Create Virtual Environment

python -m venv .venv
source .venv/bin/activate

Windows:

.venv\Scripts\activate


⸻

Install Dependencies

pip install -r requirements.txt


⸻

Example Search

Input:

Platforms that improve patient experience

Output:

Rank	Company	Similarity	Radar	Label
1	Clearwave	0.764	92	CANDIDATE
2	MedFlow	0.742	84	CANDIDATE
3	CareCX	0.721	76	REVIEW


⸻

Technologies Used

Backend
	•	Python
	•	Flask
	•	Pandas
	•	NumPy
	•	Scikit-learn

AI / NLP
	•	Cohere Embeddings API
	•	Cosine Similarity
	•	Semantic Ranking

Frontend
	•	HTML5
	•	CSS3
	•	Jinja2

⸻

Why Cohere Was Used

Originally, local embeddings were generated using:

sentence-transformers/all-MiniLM-L6-v2

However, Render’s free hosting plan had RAM constraints that caused deployment instability.

To solve this, embeddings were migrated to Cohere API, which provided:
	•	lower memory usage
	•	stable hosting
	•	faster deployment
	•	production-ready API workflow

⸻

Hosting

Currently deployed on:
	•	Render

Can also be deployed to:
	•	Railway
	•	PythonAnywhere
	•	VPS
	•	Docker

⸻

Interview Value

This project demonstrates practical experience in:

Data Engineering
	•	ETL pipelines
	•	dataset normalization
	•	quality control

Machine Learning
	•	embeddings
	•	semantic search
	•	ranking systems

Product Thinking
	•	lead prioritization
	•	explainable scoring
	•	search UX

Software Engineering
	•	Flask deployment
	•	API integration
	•	modular architecture

⸻

AI Assistance Disclosure

AI tools were used during development for:
	•	coding acceleration
	•	debugging
	•	prompt iteration
	•	UI improvements

All final logic, testing, deployment, and integration decisions were manually reviewed.

See:

AI_USAGE.md
