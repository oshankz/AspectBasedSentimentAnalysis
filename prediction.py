"""Prediction helpers for aspect-based sentiment analysis."""

from typing import Dict, List

from aspect_extraction import detect_aspects
from data_preprocessing import preprocess_text

SENTIMENT_SCORE = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}


def predict_feedback(model, feedback: str) -> Dict:
    """Predict sentiment for every detected aspect in a feedback string."""
    clean_text = preprocess_text(feedback)
    aspects: List[str] = detect_aspects(feedback)

    # One global sentiment prediction is produced from full text,
    # then mapped to each detected aspect for ABSA output.
    sentiment = model.predict([clean_text])[0]

    # Use model confidence (max probability) as intensity for score scaling.
    probabilities = model.predict_proba([clean_text])[0]
    confidence = float(max(probabilities))
    base_score = SENTIMENT_SCORE.get(sentiment, 0.0)

    aspect_results = []
    for aspect in aspects:
        aspect_results.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "sentiment_score": round(base_score * confidence, 3),
            }
        )

    return {
        "feedback": feedback,
        "detected_aspects": aspects,
        "overall_sentiment": sentiment,
        "confidence": round(confidence, 3),
        "aspect_results": aspect_results,
    }
