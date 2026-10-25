"""GetReview AI — Streamlit interface for the original Google Colab project."""
import io
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from engine import clean_and_prepare_dataset, load_models, analyze_reviews, make_recommender, MIN_MENTIONS

st.set_page_config(page_title="GetReview AI", page_icon="🏨", layout="wide")
st.title("🏨 GetReview AI")
st.caption("Turn hotel guest feedback into actionable insights — multilingual AI-powered review intelligence")
st.info("Upload a CSV or Excel file with the original notebook columns: name, reviews.text, reviews.rating, reviews.doRecommend, categories, province. No guest data is stored intentionally by this app; avoid uploading personal or confidential information to a public demo.")

@st.cache_resource(show_spinner="Loading the three Hugging Face models (first run may take several minutes)…")
def cached_models():
    return load_models()

uploaded = st.file_uploader("Upload hotel reviews", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.markdown("**How it works:** Upload reviews → Analyze sentiment & topics → Explore hotel dashboard → Generate AI recommendations → Export results.")
    st.stop()
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded)
    else:
        raw = pd.read_csv(uploaded)
    required = {"name", "reviews.text"}
    if not required.issubset(raw.columns):
        st.error("Missing mandatory columns: " + ", ".join(sorted(required - set(raw.columns))))
        st.stop()
    reviews = clean_and_prepare_dataset(raw)
    reviews = reviews.dropna(subset=["hotel"]).copy()
    reviews["hotel"] = reviews["hotel"].astype(str)
    if reviews.empty:
        st.warning("No valid hotel reviews found after cleaning.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

st.success(f"{len(reviews):,} reviews ready · {reviews['hotel'].nunique()} hotels")
with st.expander("Preview cleaned data"):
    st.dataframe(reviews.head(30), use_container_width=True)
max_reviews = st.number_input("Maximum reviews to analyze (for a faster public demo)", min_value=1, max_value=len(reviews), value=min(100, len(reviews)), step=1)
if st.button("🔍 Analyze reviews", type="primary"):
    try:
        with st.spinner("Loading AI models…"):
            sentiment_model, topic_model, tokenizer, model, device = cached_models()
        progress = st.progress(0, text="Analyzing reviews…")
        analyzed, summary = analyze_reviews(reviews.head(max_reviews), sentiment_model, topic_model, lambda p: progress.progress(min(100, int(p * 100)), text="Analyzing reviews…"))
        st.session_state.pop("report", None)
        st.session_state.pop("report_hotel", None)
        st.session_state["analyzed"] = analyzed
        st.session_state["summary"] = summary
        st.session_state["file_signature"] = (uploaded.name, uploaded.size, max_reviews)
        progress.empty()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

if st.session_state.get("file_signature") != (uploaded.name, uploaded.size, max_reviews):
    st.info("Click **Analyze reviews** to generate the dashboard.")
    st.stop()

analyzed = st.session_state["analyzed"]
summary = st.session_state["summary"]
hotel = st.selectbox("Choose a hotel", sorted(analyzed["hotel"].unique()))
hotel_reviews = analyzed[analyzed["hotel"] == hotel]
hotel_topics = summary[summary["hotel"] == hotel]

st.subheader("Hotel dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Reviews", len(hotel_reviews))
c2.metric("Negative reviews", int((hotel_reviews["sentiment"] == "negative").sum()))
c3.metric("Topics detected", len(hotel_topics))
left, right = st.columns(2)
with left:
    counts = hotel_reviews["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.plotly_chart(px.pie(counts, names="sentiment", values="count", title="Sentiment distribution"), use_container_width=True)
with right:
    if not hotel_topics.empty:
        st.plotly_chart(px.bar(hotel_topics.sort_values("negative"), x="negative", y="topic", orientation="h", title="Negative mentions by topic"), use_container_width=True)
    else:
        st.info("No topics above the semantic similarity threshold.")

st.subheader("AI recommendations")
st.caption(f"Original notebook rule: at least {MIN_MENTIONS} mentions per topic before ranking. Actions are predefined; the narrative summary is generated with FLAN-T5.")
if st.button("✨ Generate hotel report"):
    try:
        _, _, tokenizer, model, device = cached_models()
        with st.spinner("Generating evidence-based report…"):
            report = make_recommender(analyzed, summary, tokenizer, model, device)(hotel)
        st.session_state["report"] = report
        st.session_state["report_hotel"] = hotel
    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
if st.session_state.get("report_hotel") == hotel and "report" in st.session_state:
    report = st.session_state["report"]
    st.write("**AI summary:**", report["summary"])
    for rec in report["recommendations"]:
        with st.expander(f"{rec['topic']} · {rec['negative_reviews']} negative mentions", expanded=True):
            st.write("**Suggested action:**", rec["action"])
            st.write("**Evidence:**")
            for evidence in rec["evidence"]:
                st.write("•", evidence)
    st.download_button("Download report (JSON)", json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"), file_name="getreview_report.json", mime="application/json")

export = analyzed.copy()
export["topics"] = export["topics"].apply(json.dumps)
st.download_button("Download analyzed reviews (CSV)", export.to_csv(index=False).encode("utf-8-sig"), file_name="getreview_analysis.csv", mime="text/csv")
st.download_button("Download topic summary (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), file_name="getreview_topics.csv", mime="text/csv")
