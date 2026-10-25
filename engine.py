"""GetReview AI core — original Colab logic retained where possible."""
import numpy as np
import pandas as pd

def clean_and_prepare_dataset(raw_df):
    columns = ["name", "reviews.text", "reviews.rating", "reviews.doRecommend", "categories", "province", "reviews.date"]
    for col in columns:
        if col not in raw_df.columns:
            raw_df[col] = np.nan

    prepared = raw_df[columns].copy()
    prepared = prepared.dropna(subset=["reviews.text"])
    prepared["reviews.text"] = prepared["reviews.text"].str.strip()
    prepared = prepared[prepared["reviews.text"] != ""]
    prepared = prepared.drop_duplicates()
    prepared = prepared.rename(columns={
        "name": "hotel",
        "reviews.text": "review",
        "reviews.rating": "rating",
        "reviews.doRecommend": "recommended",
        "categories": "category",
        "province": "province",
        "reviews.date": "date"
    })
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return prepared.reset_index(drop=True)

TOPICS = {
    "Cleanliness": [
        "The hotel room was clean or dirty",
        "The bathroom hygiene and housekeeping",
        "The cleanliness of the hotel"
    ],
    "Staff & Service": [
        "The hotel staff were helpful or rude",
        "The quality of customer service",
        "The reception and check-in experience"
    ],
    "Room Comfort": [
        "The bed and pillows were comfortable",
        "The room was spacious or uncomfortable",
        "The quality of sleep and room amenities"
    ],
    "Location": [
        "The hotel location was convenient",
        "The distance to attractions and transport",
        "The hotel neighborhood and accessibility"
    ],
    "Food & Breakfast": [
        "The hotel breakfast was good or bad",
        "The restaurant and food quality",
        "The breakfast selection and dining experience"
    ],
    "Price & Value": [
        "The hotel was worth the price",
        "The room was expensive or affordable",
        "The value for money of the hotel"
    ],
    "Noise": [
        "The hotel room was noisy or quiet",
        "Noise from traffic or neighboring rooms",
        "The sound insulation of the hotel"
    ],
    "Facilities": [
        "The hotel pool gym and spa facilities",
        "The WiFi and internet connection",
        "The hotel equipment and maintenance"
    ]
}

SIMILARITY_THRESHOLD = 0.35

MIN_MENTIONS = 5

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOPIC_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
GENERATION_MODEL = "google/flan-t5-base"

ACTION_LIBRARY = {
    "Cleanliness": "Audit housekeeping procedures and room inspections.",
    "Staff & Service": "Review staff training and customer complaint handling.",
    "Room Comfort": "Inspect bedding, room equipment and guest comfort.",
    "Location": "Improve transport information and arrival guidance.",
    "Food & Breakfast": "Review breakfast quality, selection and service.",
    "Price & Value": "Review pricing transparency and perceived value.",
    "Noise": "Investigate noise sources and improve sound insulation.",
    "Facilities": "Inspect maintenance, WiFi and hotel facilities."
}

def load_sentiment_model():
    import torch
    from transformers import pipeline
    torch.set_num_threads(2)
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)


def load_topic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TOPIC_MODEL_NAME, device="cpu")


def load_generation_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    return tokenizer, model.to("cpu").eval(), "cpu"


def load_models():
    """API historique : conservée pour les utilisateurs du notebook."""
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from sentence_transformers import SentenceTransformer
    device = 0 if torch.cuda.is_available() else -1
    sentiment_model = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)
    topic_model = SentenceTransformer(TOPIC_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    generation_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(generation_device).eval()
    return sentiment_model, topic_model, tokenizer, model, generation_device


def analyze_reviews(reviews, sentiment_model, topic_model, progress=None):
    reviews = reviews.copy()
    texts = reviews["review"].fillna("").astype(str).tolist()
    predictions = []
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        predictions.extend(sentiment_model(texts[start:start + batch_size], truncation=True, max_length=256, batch_size=batch_size))
        if progress:
            progress(0.5 * min(start + batch_size, len(texts)) / len(texts))
    reviews["sentiment"] = [result["label"].lower() for result in predictions]
    reviews["confidence"] = [result["score"] for result in predictions]
    topic_names = list(TOPICS.keys())
    topic_descriptions = [description for topic in topic_names for description in TOPICS[topic]]
    description_topics = [topic for topic in topic_names for _ in TOPICS[topic]]
    topic_embeddings = topic_model.encode(topic_descriptions, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    review_embeddings = topic_model.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    similarities = review_embeddings @ topic_embeddings.T
    topic_scores = np.column_stack([similarities[:, [i for i,t in enumerate(description_topics) if t == topic]].max(axis=1) for topic in topic_names])
    for i, topic in enumerate(topic_names):
        reviews[f"score_{topic}"] = topic_scores[:, i]
    reviews["topics"] = [[topic_names[j] for j, score in enumerate(row) if score >= SIMILARITY_THRESHOLD] for row in topic_scores]
    reviews["topic_count"] = reviews["topics"].apply(len)
    reviews["review_id"] = np.arange(len(reviews))
    topic_data = reviews[["review_id", "hotel", "sentiment", "topics"]].explode("topics").dropna(subset=["topics"])
    if topic_data.empty:
        topic_summary = pd.DataFrame(columns=["hotel", "topic", "positive", "neutral", "negative", "total_mentions", "negative_rate"])
    else:
        topic_summary = topic_data.groupby(["hotel", "topics", "sentiment"]).size().unstack(fill_value=0).reset_index()
        for sentiment in ["positive", "neutral", "negative"]:
            if sentiment not in topic_summary.columns:
                topic_summary[sentiment] = 0
        topic_summary["total_mentions"] = topic_summary["positive"] + topic_summary["neutral"] + topic_summary["negative"]
        topic_summary["negative_rate"] = (topic_summary["negative"] / topic_summary["total_mentions"] * 100).round(1)
        topic_summary = topic_summary.rename(columns={"topics": "topic"})
    if progress:
        progress(1.0)
    return reviews, topic_summary


def make_recommender(reviews, topic_summary, tokenizer, model, device):
    import torch
    def generator(prompt, **kwargs):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        if "max_length" not in kwargs and "max_new_tokens" not in kwargs:
            kwargs["max_length"] = 256
        with torch.no_grad():
            outputs = model.generate(**inputs, **kwargs)
        return [{"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}]

    def build_hotel_profile(hotel_name, min_mentions=5):

        hotel_reviews = reviews[
            reviews["hotel"] == hotel_name
        ].copy()

        if hotel_reviews.empty:
            raise ValueError("Hotel not found")

        hotel_topics = topic_summary[
            topic_summary["hotel"] == hotel_name
        ].copy()

        total = len(hotel_reviews)

        sentiment_pct = (
            hotel_reviews["sentiment"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .to_dict()
        )

        for label in ["positive", "neutral", "negative"]:
            sentiment_pct.setdefault(label, 0.0)

        # Require enough mentions for topic ranking
        eligible = hotel_topics[
            hotel_topics["total_mentions"] >= min_mentions
        ].copy()

        strengths = eligible.sort_values(
            ["positive", "total_mentions"],
            ascending=False
        ).head(3)

        weaknesses = eligible[eligible["negative"] > 0].sort_values(
            ["negative", "negative_rate"],
            ascending=False
        ).head(3)

        return {
            "hotel": hotel_name,
            "total_reviews": total,
            "sentiment": sentiment_pct,
            "strengths": strengths.to_dict("records"),
            "weaknesses": weaknesses.to_dict("records")
        }

    def get_review_evidence(
        hotel_name,
        topic,
        sentiment="negative",
        max_reviews=3
    ):
        subset = reviews[
            (reviews["hotel"] == hotel_name) &
            (reviews["sentiment"] == sentiment)
        ].copy()

        subset = subset[
            subset["topics"].apply(
                lambda topics: topic in topics
            )
        ]

        # Prefer reviews with a high topic similarity score
        score_column = f"score_{topic}"

        if score_column in subset.columns:
            subset = subset.sort_values(
                score_column, ascending=False
            )

        subset = subset.drop_duplicates(subset=["review"])

        return subset["review"].head(max_reviews).tolist()

    def generate_hotel_recommendations(hotel_name):
        profile = build_hotel_profile(hotel_name)
        strengths = profile["strengths"]
        weaknesses = profile["weaknesses"]

        if not weaknesses:
            return {
                "hotel": hotel_name,
                "summary": "Insufficient topic evidence for recommendations.",
                "recommendations": []
            }

        recommendations = []
        for problem in weaknesses[:3]:
            topic = problem["topic"]
            evidence = get_review_evidence(
                hotel_name,
                topic,
                sentiment="negative",
                max_reviews=2
            )

            action = ACTION_LIBRARY.get(
                topic,
                "Investigate the recurring customer feedback."
            )

            recommendations.append({
                "topic": topic,
                "negative_reviews": int(problem["negative"]),
                "negative_rate": float(problem["negative_rate"]),
                "action": action,
                "evidence": evidence
            })

        facts = (
            f"Hotel: {hotel_name}\n"
            f"Total reviews: {profile['total_reviews']}\n"
            f"Positive: {profile['sentiment']['positive']}%\n"
            f"Neutral: {profile['sentiment']['neutral']}%\n"
            f"Negative: {profile['sentiment']['negative']}%\n"
            f"Positive themes: "
            f"{', '.join(x['topic'] for x in strengths)}\n"
            f"Main problem themes: "
            f"{', '.join(x['topic'] for x in weaknesses)}"
        )

        prompt = (
            "Write a short, professional English summary "
            "of the following hotel customer feedback. "
            "Use only the supplied facts. "
            "Do not invent statistics or customer complaints. "
            "Mention strengths and areas needing improvement.\n\n"
            + facts + "\n\nSummary:"
        )

        # Suppression de 'truncation' qui provoquait l'erreur avec model.generate
        result = generator(
            prompt,
            max_new_tokens=110,
            do_sample=False
        )

        summary = result[0]["generated_text"].strip()

        return {
            "hotel": hotel_name,
            "total_reviews": profile["total_reviews"],
            "sentiment": profile["sentiment"],
            "strengths": strengths,
            "summary": summary,
            "recommendations": recommendations
        }
    return generate_hotel_recommendations


def monthly_satisfaction(hotel_reviews):
    """Percentage of positive sentiment reviews per calendar month; never invent dates."""
    if "date" not in hotel_reviews.columns:
        return pd.DataFrame(columns=["month", "satisfaction", "reviews"])
    valid = hotel_reviews.dropna(subset=["date"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["month", "satisfaction", "reviews"])
    valid["month"] = valid["date"].dt.to_period("M").dt.to_timestamp()
    return (valid.groupby("month").agg(
        satisfaction=("sentiment", lambda x: round(100 * x.eq("positive").mean(), 1)),
        reviews=("sentiment", "size")
    ).reset_index().sort_values("month"))
