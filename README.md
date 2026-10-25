# GetReview AI — Streamlit deployment (v2)

GetReview AI transforms multilingual hotel guest reviews into sentiment insights, topic detection, and evidence-based hotel recommendations. Original Colab code is preserved in `original_notebook.ipynb`; the Streamlit interface is in `app.py` and the reused model logic in `engine.py`.

## Features
- Overview: hotel metrics, latest dated month satisfaction, previous-calendar-month comparison, monthly satisfaction line chart. Satisfaction = percentage of analyzed reviews classified **positive** (not an official survey score). Comparison is shown only when the immediately preceding calendar month has dated reviews. No invented dates.
- Top 3 pain points: topics with >= 5 mentions and >= 1 negative review, ranked by negative mentions then negative rate.
- AI report: maximum three recommendations, using the original `ACTION_LIBRARY`, with guest evidence and a FLAN-T5 narrative summary.
- Exports: analyzed reviews CSV, topic summary CSV, hotel report JSON.

## Input columns
Mandatory: `name` (hotel), `reviews.text` (review). Optional: `reviews.date` (necessary for monthly KPI/chart), `reviews.rating`, `reviews.doRecommend`, `categories`, `province`. Dates are parsed; invalid/missing dates are excluded from monthly results. Use anonymized/public data.

## Deploy / update
1. Unzip the release and upload **the files inside** to the root of your existing GitHub repository (replace matching files).
2. Streamlit Community Cloud will normally redeploy after GitHub changes; use **Reboot app** if needed.
3. Main file: `app.py`; dependencies: `requirements.txt`. No API keys needed.
4. Upload a small CSV and run analysis; verify the Overview, Top 3, AI report, and exports.

## Run locally
`python -m pip install -r requirements.txt` then `streamlit run app.py`. Hugging Face models download on first run; cloud memory limits may prevent successful hosting.

## Models and references
- Hugging Face `cardiffnlp/twitter-xlm-roberta-base-sentiment` (sentiment)
- Sentence Transformers `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (topics)
- Hugging Face `google/flan-t5-base` (narrative generation)
- Libraries: Python, pandas, numpy, PyTorch, transformers, sentence-transformers, Streamlit, Plotly.
- Dataset: user-supplied CSV/Excel; document its exact provenance in the assignment.

## Limitations
Monthly metrics reflect only reviews included in the chosen analysis limit and months with valid dates. They are not representative of missing data. Model-generated sentiment and summaries may be inaccurate; verify evidence. Models may exceed free hosting memory.

## Version 3 — Hotel-first selection and dark mode

Select a hotel **before** choosing the maximum number of reviews. The limit is applied only to that hotel's reviews. Changing either the hotel or the limit requires running the analysis again.

The interface now uses theme-aware CSS and a high-contrast dark palette in `.streamlit/config.toml`. For a different palette, edit that file; do not hardcode pale backgrounds in `app.py`. The original notebook and AI models are unchanged.

To update an existing GitHub repository, upload the new `app.py`, `README.md`, and `.streamlit/config.toml` (keep the `.streamlit` folder), commit, and allow Streamlit Community Cloud to redeploy.


## Version 4 — Guest Favorites and evidence cards

- Overview shows the top three positively mentioned topics, ranked by positive mention count (minimum topic evidence threshold unchanged). Expand **Read what guests liked** to see up to two authentic positive reviews per topic.
- AI Report shows authentic negative review evidence in styled, accessible cards with sentiment and topic badges. Review text is HTML-escaped; no ratings are invented.
- The original analysis models, notebook and top-three negative recommendation logic remain unchanged.


## Version 5 — Cloud resource optimization

The sentiment and topic models are loaded once on demand for the dashboard (`@st.cache_resource`). **FLAN-T5 is not loaded until Generate report is clicked**, and then is cached for subsequent report requests. This reduces initial memory use and initial loading time, but the full report still needs enough RAM to hold all three models. No quantization or model substitution is applied, so model behavior is preserved. The old `load_models()` function remains for notebook compatibility.

The deployment no longer requires `accelerate` (not used by this code). The Git ignore file prevents accidental commits of model weights, local caches, large datasets and virtual environments. `maxUploadSize = 50` MB limits accidental oversized uploads; increase it if your dataset requires more. The notebook is retained as a reproducibility reference but is not executed by Streamlit.

**Important:** This is an optimization, **not a guarantee** of running within Streamlit Community Cloud limits. If the process is terminated with out-of-memory errors after clicking Generate report, the combined model footprint is still too large for the hosting tier. Possible next steps are a smaller generation model, CPU quantization (after compatibility testing), or hosting with more RAM. Do not delete the notebook or Hugging Face source references to fix RAM usage; those files are not what consumes model inference RAM.

**Deployment update:** Upload `app.py`, `engine.py`, `requirements.txt`, `.gitignore`, `README.md`, and `.streamlit/config.toml` to the existing GitHub repository, keeping the folder hierarchy. Commit, wait for automatic redeployment, and test both dashboard and AI report.
