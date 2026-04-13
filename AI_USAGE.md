# AI_USAGE.md

## AI Usage Disclosure

## Resumen general

Durante el desarrollo del proyecto **Mini Radar + Semantic Company Search** se utilizaron herramientas de inteligencia artificial como apoyo técnico para acelerar el desarrollo, estructurar módulos del sistema, mejorar código existente y optimizar la interfaz visual.  

El uso de IA se enfocó en tareas de productividad, generación inicial de código base, refactorización y mejoras progresivas. Todas las decisiones finales de arquitectura, lógica del negocio, validación funcional, pruebas, integración entre módulos y despliegue fueron realizadas manualmente por la desarrolladora.

## Uso de prompts en inglés

Los prompts fueron redactados en inglés debido a que herramientas como Cursor, modelos de programación y asistentes de código suelen ofrecer mejores resultados técnicos, mayor precisión sintáctica y respuestas más consistentes cuando las instrucciones se formulan en inglés, especialmente para generación de código en Python, Flask, HTML y CSS.

---

# Prompts utilizados y finalidad

---

## 1. Prompt de limpieza inicial de datos

**Prompt:**

Create a complete Python script that cleans a dataset of companies from an Excel file.  
Requirements:  
1. Load the first Excel file found inside a "data" folder.  
2. Use pandas to read the file.  
3. Print:  
* file name  
* shape of dataset  
* list of columns  
4. Automatically detect the following columns (case-insensitive):  
* company name  
* description  
* website  
* sector / industry  
5. Clean the data:  
* normalize text (lowercase, strip spaces)  
* remove duplicates  
* handle missing values safely  
* clean URLs (add https if missing and validate basic structure)  
6. Create a new column called "search_text" combining:  
name + sector + description  
7. Add extra features:  
* has_website (1 if exists, else 0)  
* description_length (number of characters)  
8. Ensure an "outputs" folder exists using pathlib (create it if needed).  
9. Save the cleaned dataset as:  
outputs/cleaned_companies.csv  
10. Print:  
* cleaned dataset shape  
* confirmation message with file path  

Technical requirements:  
* use pathlib  
* modular code (helper functions like clean_text, clean_url)  
* clear structure  
* no hardcoded column names (must detect automatically)  
* script must run with:  
python src/cleaning.py  

Make the code robust and production-like, but simple enough to understand.

**Uso y justificación:**  
Se utilizó para construir el módulo inicial de limpieza de datos, indispensable para transformar la base original en Excel en un dataset estructurado y utilizable. Permitió automatizar detección de columnas, normalización de texto y preparación de variables necesarias para búsqueda semántica.

---

## 2. Prompt de refactorización del cleaning

**Prompt:**

Refactor this script to make it more robust and aligned with the original requirements.  
Please improve it with these changes:  
1. Replace os with pathlib.  
2. Save the output file as: outputs/cleaned_companies.csv  
3. Standardize the final dataset into these columns:  
* name  
* description  
* website  
* sector  
* search_text  
* has_website  
* description_length  
4. Make sure the code works even if some columns are not detected.  
5. Improve clean_url...  
6. Improve duplicate removal...  
7. Keep useful logs...  
8. Keep the code modular, readable, and production-style but simple.  
9. Make sure it runs with: python src/cleaning.py

**Uso y justificación:**  
Se utilizó para profesionalizar el script inicial, mejorar robustez, estandarización y tolerancia a errores. Fue importante para simular prácticas reales de ingeniería de datos.

---

## 3. Prompt para dataset en español

**Prompt:**

The Excel dataset is in Spanish, not English.  
Update the column detection logic so it correctly maps these columns:  
* name -> Empresa  
* description -> Descripción  
* website -> Página Web  
* sector -> Mercados  

Also consider these optional useful columns for future enrichment:  
* Oportunidades  
* Sede  
* Alianzas

**Uso y justificación:**  
Se utilizó para adaptar el sistema a la estructura real del archivo entregado. Permitió detectar correctamente columnas con nombres en español y acentos, evitando errores de mapeo.

---

## 4. Prompt para embeddings

**Prompt:**

Create a Python module called embeddings.py for a semantic search prototype.
Requirements:
1. Load the cleaned dataset from:outputs/cleaned_companies.csv
2. Use sentence-transformers with a lightweight model suitable for semantic similarity, such as:all-MiniLM-L6-v2
3. Generate embeddings for the "search_text" column.
4. Save the embeddings efficiently to disk inside outputs/, for example:outputs/company_embeddings.npy
5. Also save a copy of the cleaned dataframe with consistent indexing if needed.
6. Include modular functions:
    * load_cleaned_data()
    * load_model()
    * generate_embeddings()
    * save_embeddings()
7. Add useful logs:
    * number of rows
    * embedding shape
    * file paths saved
8. The module must be runnable directly with:python3 src/embeddings.py
Keep the code simple, clean, and production-style.

**Uso y justificación:**  
Se empleó para crear el módulo que transforma texto empresarial en vectores semánticos mediante modelos NLP. Esta etapa es el núcleo técnico del buscador, ya que permite comparar significado entre consultas y empresas.

---

## 5. Prompt para search.py

**Prompt:**

Create a Python module called search.py for semantic search over companies.
Requirements:
1. Load:
    * outputs/cleaned_companies.csv
    * outputs/company_embeddings.npy
2. Use sentence-transformers with the same model used for embeddings.
3. Implement a function that:
    * receives a natural language query
    * generates an embedding for the query
    * computes cosine similarity against company embeddings
    * returns the top N most relevant companies
4. Return for each result:
    * name
    * description
    * website
    * sector
    * similarity score
5. Include a test block under:if name == "main":that runs a sample query like:"plataformas para mejorar la atención al paciente"
6. Print the top results in a readable way.
Keep the code simple, modular, and easy to connect later to Flask.

**Uso y justificación:**  
Se utilizó para construir el motor de búsqueda principal. Permitió recibir consultas en lenguaje natural, calcular similitud coseno y devolver resultados relevantes ordenados.

---

## 6. Prompt para corrección de similitud

**Prompt:**

Fix the semantic similarity computation in this file.
Problem:The current code fails with:ValueError: cannot select an axis to squeeze out which has size not equal to one
Requirements:
1. Make sure the query embedding is always converted into a 1D vector before similarity calculation.
2. Compute cosine similarity in a robust way between:
    * company_embeddings with shape (n, d)
    * query_embedding with shape (d,)
3. Do not use fragile squeeze(axis=...) logic.
4. Return a 1D similarity array of length n.
5. Keep the code simple and production-style.
If useful, normalize both vectors and compute cosine similarity explicitly.


**Uso y justificación:**  
Se utilizó para resolver un error técnico relacionado con dimensiones vectoriales durante el cálculo de similitud. Permitió estabilizar el algoritmo de búsqueda.

---

## 7. Prompt scoring.py

**Prompt:**

Create a Python module called scoring.py for a strategic radar score.
Requirements:
1. Load the cleaned dataset from:outputs/cleaned_companies.csv
2. Build a radar score from 0 to 100 using 3 components:
A. Industry alignment (35 points)Give points if the company sector or search_text matches any of these target industries:
* telecomunicaciones
* banca
* fintech
* salud
* educación
* medios
* entretenimiento
* música
* agricultura
* minería
* energía
B. Impact alignment (35 points)Give points if the company description or search_text suggests alignment with one or more of these business goals:
* vender más
* mejorar experiencia del cliente
* optimizar operaciones / reducir costos / automatizar procesos
Use simple keyword-based logic for now.
C. Data quality (30 points)Score based on:
* has_website
* description_length
* sector not empty
* search_text not empty
1. Add these new columns:
* industry_score
* impact_score
* data_quality_score
* radar_score
* radar_label
1. Assign radar_label as:
* CANDIDATE if score > 80
* REVIEW if score between 40 and 79
* DISCARD if score < 40
1. Save the enriched dataset as:outputs/scored_companies.csv
2. Print:
* number of rows
* average radar score
* label distribution
* save path
1. Keep the code modular and simple, with helper functions for each score component.
Make it production-style but easy to explain in an interview.

**Uso y justificación:**  
Se empleó para agregar una segunda capa analítica al proyecto: además de encontrar empresas relevantes, también clasificarlas estratégicamente mediante un radar score.

---

## 8. Prompt para mejorar scoring

**Prompt:**

The current radar score is too saturated at 100 for many companies.
Refactor the scoring logic to make it more discriminative and realistic.
Requirements:
1. Industry alignment (max 35 points)
* Give full points only if the company clearly belongs to one of the target industries.
* If the match is weak or only partial, assign fewer points.
1. Impact alignment (max 35 points)
* Separate the score into 3 subcategories:
    * revenue / selling more
    * customer experience
    * operational efficiency
* Give partial points depending on how many categories are matched.
* Avoid giving full points too easily just because one keyword appears.
1. Data quality (max 30 points)
* Score progressively, not all-or-nothing:
    * website valid
    * description length
    * sector available
    * search_text richness
1. Add an explainability column called:score_reasonthat briefly explains why the company received its score.
2. Keep these output columns:
* industry_score
* impact_score
* data_quality_score
* radar_score
* radar_label
* score_reason
1. Save again to:outputs/scored_companies.csv
Make the scoring more balanced, realistic, and easy to explain in an interview.

**Uso y justificación:**  
Se utilizó para mejorar la calidad del modelo de scoring, evitando resultados irreales y generando puntuaciones más diferenciadas y defendibles en contexto empresarial.

---

## 9. Prompt app.py

**Prompt:**

Create a simple Flask app for the mini radar project.
Requirements:
1. Load data from:
    * outputs/scored_companies.csv
    * outputs/company_embeddings.npy
2. Reuse semantic search logic with sentence-transformers and cosine similarity.
3. Create two routes:
    * "/" for the main page
    * "/search" to handle search queries
4. On the main page, include:
    * a text input for natural language search
    * a submit button
5. On search:
    * receive the query
    * compute semantic similarity
    * return the top 10 matching companies
6. For each result, display:
    * name
    * sector
    * website
    * radar_score
    * radar_label
    * similarity
    * short description
    * score_reason
7. Keep the Flask app simple and readable.
8. Do not overengineer.
9. Assume the template file is templates/index.html
10. The app must run with:python3 app.py
Make it easy to explain in an interview.


**Uso y justificación:**  
Se utilizó para desarrollar el backend web del proyecto usando Flask, integrando datos procesados, embeddings y búsquedas en una aplicación funcional.

---

## 10. Prompt index.html

**Prompt:**

Create a premium modern HTML template for a Flask app called:
Mini Radar + Semantic Company Search
Design goals:
* visually impressive
* modern startup / AI product landing page
* not flat or plain
* elegant layout with strong hierarchy
* premium look similar to a modern portfolio / landing page
* polished enough that it does not look like raw HTML
Requirements:
1. Use Jinja templating compatible with Flask.
2. Include:
    * hero section with large title
    * subtitle explaining the product
    * centered search bar with submit button
3. If results exist, display them in elegant result cards.
4. Each result card must show:
    * name
    * sector
    * website link
    * radar_score
    * radar_label
    * similarity
    * description
    * score_reason
5. Add proper semantic structure:
    * header
    * main
    * results section
6. Link an external stylesheet:{{ url_for('static', filename='styles.css') }}
7. Keep the HTML clean and professional.
8. Do not use JavaScript.
The page should feel like a polished AI product, not a student exercise.


**Uso y justificación:**  
Se empleó para diseñar una interfaz moderna y profesional orientada a mejorar presentación visual, experiencia del usuario e impacto durante evaluación técnica.

---

## 11. Prompt styles.css

**Prompt:**

Create a modern premium CSS file for the Flask Mini Radar app.
Design direction:
* inspired by elegant portfolio / landing pages
* modern AI startup aesthetic
* visually impactful
* not flat
* use gradients, soft shadows, glassmorphism touches, rounded corners, elegant spacing
* make the hero section feel premium
* make search bar large and central
* result cards should feel high-end and polished
* use strong typography hierarchy
* subtle transitions and hover states
* dark or dark-blue background with bright accents
* badge styles for radar labels:
    * CANDIDATE
    * REVIEW
    * DISCARD
Requirements:
1. Style:
    * body
    * header / hero
    * form
    * input
    * button
    * results grid
    * result card
    * badges
    * links
    * score blocks
    * description text
    * score_reason section
2. Make it responsive.
3. Use CSS only.
4. Make it look like a real modern product page.
5. Avoid excessive complexity, but make it visually impressive.
Return full CSS code.

**Uso y justificación:**  
Se utilizó para estilizar la aplicación con apariencia más comercial y cercana a productos reales del mercado, elevando calidad visual del prototipo.

---

## 12. Prompt mejora visual CSS

**Prompt:**

Improve this CSS so the page looks more premium and visually impressive.
Enhance:
* hero section presence
* typography contrast
* spacing and layout rhythm
* result cards depth
* glassmorphism / blur touches
* subtle gradients
* hover interactions
* badge polish
* section transitions
Do not make it flashy or messy. Keep it elegant, modern, and product-grade.



**Uso y justificación:**  
Se utilizó para iterar sobre el diseño visual final, agregando profundidad, jerarquía visual y acabados más profesionales.

---

# Supervisión humana

Todo código generado mediante IA fue revisado, corregido, probado e integrado manualmente. La lógica final del producto, conexión entre módulos, validación de resultados y decisiones técnicas fueron responsabilidad directa de la desarrolladora.

