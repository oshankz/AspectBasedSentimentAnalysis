"""Streamlit app for Aspect Based Sentiment Analysis of Student Feedback."""

from collections import Counter

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from aspect_extraction import ASPECT_KEYWORDS
from model_training import load_model, train_model
from prediction import predict_feedback

st.set_page_config(
    page_title="Aspect Based Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; color: #1f3b4d;}
    .subtitle {font-size: 1rem; color: #5b6b73; margin-bottom: 1rem;}
    .sent-card {padding: 0.8rem 1rem; border-radius: 12px; color: #ffffff; margin-bottom: 0.6rem;}
    .positive {background: #2e7d32;}
    .neutral {background: #f9a825;}
    .negative {background: #c62828;}
    .metric-box {background: #f5f7fa; padding: 1rem; border-radius: 12px; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>Aspect Based Sentiment Analysis of Student Feedback</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Identify aspects and classify sentiment using NLP + TF-IDF + Logistic Regression.</div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Load an existing model or train a new one automatically."""
    return load_model()


model = get_model()

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Single Feedback", "Dataset Analysis", "Model"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Predefined Aspects")
for aspect in ASPECT_KEYWORDS:
    st.sidebar.write(f"- {aspect}")


def sentiment_class(sentiment: str) -> str:
    if sentiment == "Positive":
        return "positive"
    if sentiment == "Negative":
        return "negative"
    return "neutral"


def render_cards(aspect_results):
    for row in aspect_results:
        css_class = sentiment_class(row["sentiment"])
        st.markdown(
            f"<div class='sent-card {css_class}'><b>{row['aspect']}</b> &nbsp;|&nbsp; "
            f"Sentiment: <b>{row['sentiment']}</b> &nbsp;|&nbsp; Score: <b>{row['sentiment_score']}</b></div>",
            unsafe_allow_html=True,
        )


if section == "Single Feedback":
    st.subheader("Analyze Individual Feedback")
    feedback = st.text_area("Enter student feedback", placeholder="Type feedback about faculty, curriculum, infrastructure, placements, or management...")

    if st.button("Analyze Feedback", type="primary"):
        if feedback.strip():
            result = predict_feedback(model, feedback)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("### Detected Aspects")
                st.write(", ".join(result["detected_aspects"]))
                st.write(f"### Overall Sentiment: **{result['overall_sentiment']}**")
                st.write(f"Confidence: **{result['confidence']}**")

            with col2:
                st.write("### Aspect-wise Sentiment")
                render_cards(result["aspect_results"])
        else:
            st.warning("Please enter feedback text before analysis.")

elif section == "Dataset Analysis":
    st.subheader("Analyze CSV Dataset")
    st.caption("Upload a CSV with at least a 'feedback' column. Optional 'sentiment' label can be included for your own reference.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Normalize possible feedback column names.
        feedback_col = None
        for col in df.columns:
            if col.lower() == "feedback":
                feedback_col = col
                break

        if feedback_col is None:
            st.error("CSV must contain a 'feedback' column.")
        else:
            if st.button("Run Dataset Analysis", type="primary"):
                results = [predict_feedback(model, text) for text in df[feedback_col].fillna("").astype(str)]

                flattened = []
                for res in results:
                    for aspect_row in res["aspect_results"]:
                        flattened.append(
                            {
                                "feedback": res["feedback"],
                                "aspect": aspect_row["aspect"],
                                "sentiment": aspect_row["sentiment"],
                                "sentiment_score": aspect_row["sentiment_score"],
                            }
                        )

                out_df = pd.DataFrame(flattened)
                st.write("### Predictions Preview")
                st.dataframe(out_df.head(20), use_container_width=True)

                # Dashboard metrics.
                total_records = len(results)
                sentiment_counts = Counter(out_df["sentiment"])
                avg_score = out_df["sentiment_score"].mean() if not out_df.empty else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Feedback", total_records)
                c2.metric("Average Sentiment Score", round(float(avg_score), 3))
                c3.metric("Detected Aspects", out_df["aspect"].nunique())

                st.write("### Overall Performance Dashboard")
                left, right = st.columns(2)

                with left:
                    st.write("#### Sentiment Distribution (Bar)")
                    bar_data = pd.DataFrame(
                        {"Sentiment": list(sentiment_counts.keys()), "Count": list(sentiment_counts.values())}
                    )
                    st.bar_chart(bar_data.set_index("Sentiment"))

                    st.write("#### Aspect Frequency")
                    st.bar_chart(out_df["aspect"].value_counts())

                with right:
                    st.write("#### Sentiment Distribution (Pie)")
                    fig, ax = plt.subplots()
                    labels = list(sentiment_counts.keys())
                    sizes = list(sentiment_counts.values())
                    color_map = {"Positive": "#2e7d32", "Neutral": "#f9a825", "Negative": "#c62828"}
                    colors = [color_map.get(label, "#607d8b") for label in labels]
                    if sizes:
                        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
                        ax.axis("equal")
                        st.pyplot(fig)
                    else:
                        st.info("No sentiment data available for pie chart.")

                    st.write("#### Word Cloud")
                    all_feedback = " ".join(df[feedback_col].fillna("").astype(str).tolist())
                    if all_feedback.strip():
                        wc = WordCloud(width=900, height=400, background_color="white", colormap="viridis").generate(all_feedback)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                        ax_wc.imshow(wc, interpolation="bilinear")
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info("Not enough text for a word cloud.")

else:
    st.subheader("Model Management")
    st.write("Train / retrain the Logistic Regression model.")

    train_upload = st.file_uploader("Optional training CSV (columns: feedback, sentiment)", type=["csv"], key="train_csv")
    if st.button("Train Model", type="primary"):
        training_path = ""
        if train_upload is not None:
            training_path = "uploaded_training_data.csv"
            with open(training_path, "wb") as f:
                f.write(train_upload.getbuffer())

        trained_model, metrics = train_model(training_path)
        st.success("Model trained and saved to sentiment_model.pkl")
        st.write(f"Accuracy: **{metrics['accuracy']:.3f}**")
        st.json(metrics["report"])
        st.cache_resource.clear()
        model = trained_model
