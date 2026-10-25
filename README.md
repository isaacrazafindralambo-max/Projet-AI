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
