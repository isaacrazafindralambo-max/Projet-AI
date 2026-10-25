"""GetReview AI — Streamlit web application, original Colab models and logic."""
import hashlib
import gc
import torch
import json
import html
import pandas as pd
import plotly.express as px
import streamlit as st
from engine import clean_and_prepare_dataset, load_sentiment_model, load_topic_model, load_generation_model, analyze_reviews, make_recommender, monthly_satisfaction, MIN_MENTIONS

st.set_page_config(page_title="GetReview AI", page_icon="🏨", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
/* Streamlit theme-aware styling: readable in both light and dark modes. */
.block-container {max-width:1250px;padding-top:1.7rem;padding-bottom:3rem;}
[data-testid="stMetric"] {
  background:var(--secondary-background-color);
  border:1px solid color-mix(in srgb, var(--text-color) 20%, transparent);
  border-radius:14px;padding:18px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {color:var(--text-color);}
[data-testid="stSidebar"] {
  background:var(--secondary-background-color);
  border-right:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h2 {color:var(--text-color);}
[data-testid="stTabs"] button {color:var(--text-color);font-weight:600;}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:color-mix(in srgb, var(--text-color) 20%, transparent);
}
[data-testid="stCaptionContainer"] {opacity:.9;}
h1,h2,h3 {letter-spacing:-.025em;}
.review-card {border:1px solid color-mix(in srgb, var(--text-color) 20%, transparent);border-radius:16px;padding:1.1rem 1.3rem;margin:.65rem 0 1rem;background:var(--secondary-background-color);}
.review-card .review-quote {color:var(--text-color);font-size:1.02rem;line-height:1.65;margin:.8rem 0;overflow-wrap:anywhere;}
.review-card .review-top {display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;}
.review-card .review-tag {display:inline-block;border:1px solid color-mix(in srgb, var(--text-color) 20%, transparent);padding:.2rem .65rem;border-radius:999px;font-size:.74rem;font-weight:700;letter-spacing:.03em;}
.review-card .review-positive {background:rgba(16,185,129,.13);color:#10b981;border-color:rgba(16,185,129,.35);}
.review-card .review-negative {background:rgba(239,68,68,.13);color:#fb7185;border-color:rgba(239,68,68,.35);}
.review-card .review-topic {background:color-mix(in srgb, var(--text-color) 7%, transparent);color:var(--text-color);}
.favorites-header {margin-top:1.4rem;}

</style>""", unsafe_allow_html=True)

def review_card(text, sentiment, topic):
    """Render authentic, escaped review text; no fabricated ratings or stars."""
    safe_text = html.escape(str(text), quote=True)
    safe_topic = html.escape(str(topic), quote=True)
    positive = sentiment == "positive"
    badge = "review-positive" if positive else "review-negative"
    label = "POSITIVE" if positive else "NEGATIVE"
    st.markdown(f"""<div class="review-card">
      <div class="review-top"><span class="review-tag review-topic">{safe_topic}</span>
      <span class="review-tag {badge}">{label}</span></div>
      <p class="review-quote">“{safe_text}”</p>
    </div>""", unsafe_allow_html=True)


def topic_evidence(df, topic, sentiment="positive", max_reviews=2):
    subset = df.loc[df["sentiment"].eq(sentiment)].copy()
    subset = subset.loc[subset["topics"].apply(lambda ts: isinstance(ts, (list, tuple, set)) and topic in ts)]
    score_col = f"score_{topic}"
    if score_col in subset.columns:
        subset = subset.sort_values(score_col, ascending=False)
    return subset.drop_duplicates(subset=["review"])["review"].head(max_reviews).tolist()


def release_model(model):
    """Release large objects before loading the next model."""
    del model
    gc.collect()


st.title("🏨 GetReview AI")
st.caption("Hotel intelligence · Multilingual sentiment · Topic insights · Evidence-based recommendations")
with st.sidebar:
    st.header("Workspace")
    uploaded = st.file_uploader("Hotel reviews · CSV or Excel", type=["csv", "xlsx", "xls"])
    st.caption("Required: name, reviews.text · For monthly insights: reviews.date")
    st.warning("Use public or anonymized data for this demo.")

if uploaded is None:
    st.info("Start by uploading hotel reviews in the sidebar. Then run the analysis to unlock your dashboard.")
    st.markdown("**Workflow**  ·  Upload → Analyze → Overview → Pain points → AI report → Export")
    st.stop()

try:
    raw = pd.read_excel(uploaded) if uploaded.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded)
    missing = {"name", "reviews.text"} - set(raw.columns)
    if missing:
        st.error("Missing required columns: " + ", ".join(sorted(missing)))
        st.stop()
    reviews = clean_and_prepare_dataset(raw)
    reviews = reviews.dropna(subset=["hotel"]).copy()
    if reviews.empty:
        st.error("No valid hotel reviews found.")
        st.stop()
except Exception as exc:
    st.error(f"File could not be read: {exc}")
    st.stop()

file_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
with st.sidebar:
    st.markdown("### Analysis settings")
    hotel = st.selectbox("1 · Select a hotel", sorted(reviews["hotel"].unique()))
    selected_reviews = reviews.loc[reviews["hotel"] == hotel].copy()
    st.caption(f"{len(selected_reviews):,} available reviews for this hotel")
    limit = st.number_input(
        "2 · Maximum reviews to analyze", min_value=1,
        max_value=len(selected_reviews), value=min(100, len(selected_reviews))
    )
    st.caption("The limit applies only to the selected hotel. For reliable monthly trends, include reviews from multiple months.")
    analyze = st.button("🔍 Run AI analysis", type="primary", use_container_width=True)

signature = (file_id, hotel, int(limit))
if analyze:
    try:
        # Phase 1: classify reviews with only the sentiment model in RAM.
        with st.spinner("Step 1/2 · Loading sentiment model…"):
            sentiment_model = load_sentiment_model()
        texts = selected_reviews.head(limit)["review"].fillna("").astype(str).tolist()
        predictions = []
        bar = st.progress(0, text="Step 1/2 · Sentiment analysis…")
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                predictions.extend(sentiment_model(texts[start:start+8], truncation=True, max_length=256, batch_size=8))
                bar.progress(int(45 * min(start+8, len(texts))/len(texts)), text="Step 1/2 · Sentiment analysis…")
        del sentiment_model
        gc.collect()

        # Phase 2: load topic model only after the sentiment model is gone.
        with st.spinner("Step 2/2 · Loading topic model…"):
            topic_model = load_topic_model()
        class SavedSentiments:
            def __init__(self, values):
                self.values = values
                self.offset = 0
            def __call__(self, batch, **kwargs):
                n = len(batch)
                result = self.values[self.offset:self.offset+n]
                self.offset += n
                return result
        try:
            analyzed, summary = analyze_reviews(
                selected_reviews.head(limit), SavedSentiments(predictions), topic_model,
                lambda x: bar.progress(min(100, 45 + int(x*55)), text="Step 2/2 · Topic analysis…")
            )
        finally:
            del topic_model
            gc.collect()
        bar.empty()
        st.session_state.update(analyzed=analyzed, summary=summary, signature=signature)
        st.session_state.pop("report", None)
        st.session_state.pop("report_hotel", None)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

if st.session_state.get("signature") != signature:
    st.info("Click **Run AI analysis** in the sidebar to generate your dashboard.")
    with st.expander("Preview imported data"):
        st.dataframe(selected_reviews.head(20), use_container_width=True)
    st.stop()

analyzed, summary = st.session_state["analyzed"], st.session_state["summary"]
with st.sidebar:
    st.caption("Satisfaction = share of positive AI-classified reviews (0–100%).")
hotel_reviews = analyzed[analyzed["hotel"] == hotel].copy()
hotel_topics = summary[summary["hotel"] == hotel].copy()
monthly = monthly_satisfaction(hotel_reviews)

tab_overview, tab_pain, tab_report, tab_data = st.tabs(["📊 Overview", "🎯 Pain points", "✨ AI report", "⬇️ Exports"])
with tab_overview:
    st.subheader(hotel)
    a,b,c,d = st.columns(4)
    a.metric("Reviews analyzed", len(hotel_reviews))
    b.metric("Positive reviews", f"{100*hotel_reviews['sentiment'].eq('positive').mean():.1f}%")
    c.metric("Negative reviews", int(hotel_reviews['sentiment'].eq('negative').sum()))
    d.metric("Topics detected", len(hotel_topics))
    st.markdown("#### Monthly satisfaction")
    if monthly.empty:
        st.info("No valid review dates. Include the **reviews.date** column to display monthly satisfaction and trends.")
    else:
        latest = monthly.iloc[-1]
        prev_month = latest["month"] - pd.DateOffset(months=1)
        previous = monthly[monthly["month"] == prev_month]
        x,y = st.columns([1,2])
        with x:
            if previous.empty:
                st.metric(f"Latest month · {latest['month']:%b %Y}", f"{latest['satisfaction']:.1f}%")
                st.caption("No reviews for the immediately preceding month; comparison unavailable.")
            else:
                st.metric(f"Latest month · {latest['month']:%b %Y}", f"{latest['satisfaction']:.1f}%", f"{latest['satisfaction']-previous.iloc[0]['satisfaction']:+.1f} pp")
                st.caption(f"Previous month: {previous.iloc[0]['satisfaction']:.1f}% · Based on {int(latest['reviews'])} reviews this month")
        with y:
            fig = px.line(monthly, x="month", y="satisfaction", markers=True, hover_data=["reviews"], labels={"month":"Month", "satisfaction":"Positive reviews (%)", "reviews":"Reviews"})
            fig.update_yaxes(range=[0,100], gridcolor="rgba(128,128,128,0.25)")
            fig.update_layout(margin=dict(l=0,r=10,t=12,b=0),height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="rgba(155,155,155,1)")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Latest month with dated reviews, not necessarily the current calendar month. Only analyzed reviews are included. pp = percentage points.")
    left,right = st.columns(2)
    with left:
        counts=hotel_reviews['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
        st.plotly_chart(px.pie(counts, names='sentiment', values='count', title='Sentiment distribution', hole=.55), use_container_width=True)
    with right:
        if not hotel_topics.empty:
            st.plotly_chart(px.bar(hotel_topics.sort_values('negative'),x='negative',y='topic',orientation='h',title='Negative mentions by topic'),use_container_width=True)
        else:
            st.info("No topics detected above the similarity threshold.")
    st.markdown("#### 💚 Guest Favorites")
    st.caption("Top 3 most frequently praised topics among AI-classified positive reviews. Multiple topics can be detected in the same review.")
    favorites = hotel_topics.loc[(hotel_topics["total_mentions"] >= MIN_MENTIONS) & (hotel_topics["positive"] > 0)].sort_values(["positive", "total_mentions"], ascending=False).head(3)
    if favorites.empty:
        st.info("Not enough positive topic mentions to identify guest favorites in this sample.")
    else:
        fav_cols = st.columns(len(favorites))
        for i, (_, fav) in enumerate(favorites.iterrows()):
            with fav_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**#{i+1} · {fav['topic']}**")
                    st.metric("Positive mentions", int(fav["positive"]))
                    st.caption(f"{int(fav['total_mentions'])} total topic mentions")
        with st.expander("💬 Read what guests liked", expanded=False):
            for _, fav in favorites.iterrows():
                for excerpt in topic_evidence(hotel_reviews, fav["topic"], "positive", 2):
                    review_card(excerpt, "positive", fav["topic"])
with tab_pain:
    st.subheader("Top 3 pain points")
    eligible = hotel_topics[(hotel_topics['total_mentions'] >= MIN_MENTIONS) & (hotel_topics['negative'] > 0)].sort_values(['negative','negative_rate'],ascending=False).head(3)
    if eligible.empty:
        st.info(f"No negative topics with at least {MIN_MENTIONS} mentions in this hotel sample.")
    else:
        for rank,(_,item) in enumerate(eligible.iterrows(),1):
            with st.container(border=True):
                st.markdown(f"**#{rank} · {item['topic']}**")
                u,v=st.columns(2)
                u.metric("Negative mentions",int(item['negative']))
                v.metric("Negative rate",f"{item['negative_rate']:.1f}%")
    st.caption(f"Ranking: negative mentions, then negative rate · Minimum {MIN_MENTIONS} total mentions per topic.")
with tab_report:
    st.subheader("AI hotel report")
    st.caption("Exactly the top 3 eligible negative pain points at most. Operational actions come from the original action library; FLAN-T5 generates the narrative.")
    if st.button("✨ Generate report",type="primary"):
        try:
            with st.spinner("Loading FLAN-T5 for this report only…"):
                tokenizer, model, device = load_generation_model()
            try:
                with st.spinner("Generating report…"):
                    report = make_recommender(analyzed,summary,tokenizer,model,device)(hotel)
            finally:
                del tokenizer, model
                gc.collect()
            st.session_state['report']=report
            st.session_state['report_hotel']=hotel
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
    if st.session_state.get('report_hotel')==hotel and 'report' in st.session_state:
        report=st.session_state['report']
        st.markdown("#### Executive summary")
        st.write(report['summary'])
        for idx,rec in enumerate(report['recommendations'][:3],1):
            with st.expander(f"Priority {idx} · {rec['topic']} · {rec['negative_reviews']} negative mentions",expanded=True):
                st.markdown("**Recommended action**")
                st.write(rec['action'])
                st.markdown("**Guest evidence**")
                for evidence in rec['evidence']:
                    review_card(evidence, 'negative', rec['topic'])
        st.download_button("Download AI report · JSON",json.dumps(report,indent=2,ensure_ascii=False).encode(),file_name='getreview_report.json',mime='application/json')
with tab_data:
    st.subheader("Export results")
    exported=analyzed.copy()
    exported['topics']=exported['topics'].apply(json.dumps)
    st.download_button("Download analyzed reviews · CSV",exported.to_csv(index=False).encode('utf-8-sig'),file_name='getreview_analysis.csv',mime='text/csv')
    st.download_button("Download topic summary · CSV",summary.to_csv(index=False).encode('utf-8-sig'),file_name='getreview_topics.csv',mime='text/csv')
    st.dataframe(hotel_reviews[['hotel','date','review','sentiment','topics']].head(100).assign(topics=lambda d:d.topics.astype(str)),use_container_width=True)
