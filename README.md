# GetReview AI — Hotel Review Intelligence

Streamlit deployment of the original Google Colab notebook. The core cleaning function, hotel topics, similarity threshold, recommendation actions, and report-building functions are reused from the original notebook.

## Deploy a public URL
1. Create a **public GitHub repository** and upload `app.py`, `engine.py`, `requirements.txt`, and optionally the original `.ipynb` and documentation. Do **not** upload guest datasets or confidential reviews.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub.
3. Choose **Create app → Yup, I have an app**, select your repository, branch `main`, and entry point `app.py`.
4. Click **Deploy**. The platform will install `requirements.txt` and generate a shareable `*.streamlit.app` URL.
5. Upload a CSV or Excel file and press **Analyze reviews**. First launch downloads 3 public Hugging Face models and can take several minutes. Memory limits or cold starts may prevent deployment on free hosting; if so, use a larger hosted instance or reduce model memory usage.

## Run locally
```bash
python -m venv .venv
# Activate the virtual environment for your OS
pip install -r requirements.txt
streamlit run app.py
```

## Expected input
CSV/XLSX with at least `name` (hotel) and `reviews.text` (review), plus optional `reviews.rating`, `reviews.doRecommend`, `categories`, `province`.

## Reproducibility
- Source: `Projet_AI_2-2(2).ipynb` (included).
- Model identifiers: `cardiffnlp/twitter-xlm-roberta-base-sentiment`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `google/flan-t5-base`.
- Python packages and compatible version ranges: `requirements.txt`.
- Topic similarity threshold: 0.35; minimum mentions for recommendations: 5.
- Processing is limited to the first 100 reviews by default, adjustable in the interface. This is a **demo performance limit**, not a feature of the original notebook.
- To reproduce a specific run exactly, record Python version, resolved package versions (`pip freeze`) and downloaded model revisions. Version ranges alone do not ensure bit-for-bit reproducibility.

## Limitations
- Sentiment/topic detection is model-based and may be inaccurate.
- The topic matcher does not inherently distinguish praise from complaints; negative mentions are filtered using sentiment predictions.
- The actions come from the original predefined `ACTION_LIBRARY`; FLAN-T5 generates the narrative summary only.
- The app stores results in Streamlit session memory, not a database. Public cloud services still process uploaded data; do not use sensitive personal data.
- Free cloud compute may be insufficient for three loaded models. Deployment is prepared, **not verified online**.

## Sources
- Original project notebook, authored by the student.
- Hugging Face model repositories (identifiers above).
- Streamlit, Transformers, Sentence Transformers, PyTorch, pandas, NumPy, Plotly, SentencePiece, Accelerate, openpyxl, xlrd (package documentation and licenses).
- The source of any real hotel review dataset must be documented separately by the student; the notebook does not establish a dataset license.
