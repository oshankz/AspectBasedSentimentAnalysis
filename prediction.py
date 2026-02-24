"""
prediction.py
-------------
Handles end-to-end prediction for aspect-based sentiment analysis.
Combines preprocessing, aspect extraction, and model inference.
"""

import os
import numpy as np
from data_preprocessing import preprocess
from aspect_extraction import extract_aspects
from model_training import load_model, train_model

# Paths for saved model artifacts
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Sentiment to numeric score mapping
SENTIMENT_SCORES = {
    "Positive": 1.0,
    "Neutral": 0.0,
    "Negative": -1.0
}

# Color codes for display
SENTIMENT_COLORS = {
    "Positive": "#2ecc71",   # Green
    "Neutral":  "#f1c40f",   # Yellow
    "Negative": "#e74c3c"    # Red
}

# Emoji indicators
SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😞"
}


def get_model_and_vectorizer():
    """
    Load model from disk; train fresh if not found.

    Returns:
        tuple: (model, vectorizer)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return load_model(MODEL_PATH, VECTORIZER_PATH)
    else:
        print("No saved model found. Training now...")
        train_model()
        return load_model(MODEL_PATH, VECTORIZER_PATH)


# ─── Sentiment word lists for short-text fallback ────────────────────────────
POSITIVE_WORDS = {
    # Standard positive
    "nice", "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "superb", "outstanding", "brilliant", "awesome", "best", "love", "loved",
    "happy", "perfect", "beautiful", "clean", "modern", "helpful", "supportive",
    "knowledgeable", "talented", "efficient", "effective", "impressive",
    "top", "well", "better", "positive", "recommend", "enjoy", "enjoyed",
    "useful", "strong", "smart", "innovative", "quality", "satisfied",
    "satisfying", "rewarding", "exceptional", "remarkable", "splendid",
    # Slang / informal positive
    "lit", "banger", "fire", "solid", "dope", "clutch", "smooth",
    "decent", "nailed", "killed", "goated", "blessed", "legendary",
    "insane", "unreal", "phenomenal", "stellar", "mint", "crisp", "ace"
}

NEGATIVE_WORDS = {
    # Standard negative
    "bad", "poor", "terrible", "awful", "horrible", "worst", "useless",
    "pathetic", "dirty", "broken", "outdated", "slow", "boring",
    "rude", "arrogant", "corrupt", "rigid", "unhelpful", "disappointing",
    "disappointed", "frustrating", "frustrated", "waste", "hate", "hated",
    "ugly", "disgusting", "unhygienic", "overcrowded", "disorganized",
    "irresponsible", "careless", "negligent", "incompetent", "inadequate",
    # Slang / informal negative
    "trash", "mid", "weak", "buggy", "messy", "painful", "exhausting",
    "stressful", "draining", "confusing", "chaotic", "pointless",
    "unbearable", "ridiculous", "nightmare", "sucks", "horrible",
    "atrocious", "dreadful", "abysmal", "diabolical", "dismal"
}

NEUTRAL_WORDS = {
    # Neutral / ambiguous slang — should NOT be classified as pos or neg alone
    "okay", "fine", "average", "manageable", "fair", "acceptable",
    "normal", "standard", "typical", "alright", "meh", "moderate",
    "mediocre", "passable", "adequate", "ordinary", "so-so"
}


# ── Sarcasm/contrast pattern detector ───────────────────────────────────────
import re as _re

SARCASM_PATTERNS = [
    # Praise followed by absurd qualifier (works perfectly WHEN...)
    _re.compile(r'(great|love|amazing|excellent|wonderful|fantastic|brilliant|superb).{0,40}(when|if|except|unless|only|but|however)', _re.I),
    # Positive opener + negative ender
    _re.compile(r'(great|love|amazing|perfect|excellent).{0,60}(broken|useless|pathetic|terrible|awful|horrible|waste|never|nothing|zero)', _re.I),
    # Sarcastic qualifiers
    _re.compile(r'(if your goal is|prepares you for|teaches you|builds character|rich tradition of|consistently)', _re.I),
    # Museum / outdated absurdity
    _re.compile(r'(belong in a museum|from the 90s|from the 1990s|stopped existing|do not exist|never use)', _re.I),
    # Emotionally loaded dark phrases
    _re.compile(r'(question my life|life choices|destroying us|break.*spirit|feel.*hopeless|nostalgic for freedom)', _re.I),
]

CONTRAST_PATTERN = _re.compile(
    r'(but|however|although|though|yet|despite|unfortunately)', _re.I
)

def detect_sarcasm_or_contrast(text: str) -> bool:
    """
    Returns True if the text likely contains sarcasm, dark humour,
    or a strong contrast (positive + negative in same sentence).
    """
    for pattern in SARCASM_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Multi-word phrase lookups ─────────────────────────────────────────────
POSITIVE_PHRASES = {
    "not bad", "top notch", "on point", "worth it", "great stuff",
    "loved it", "works well", "super helpful", "amazing experience",
    "highly recommend", "no complaints", "actually impressive",
    "surprisingly good", "really good", "so good", "nailed it",
    "pretty great", "absolutely loved"
}

NEUTRAL_PHRASES = {
    "not bad", "not great", "could be better", "decent enough",
    "nothing special", "does the job", "so so", "kind of okay",
    "not bad not great", "okay i guess", "it is what it is",
    "meets expectations", "gets the work done", "pretty average",
    "somewhere in the middle", "neither good nor bad"
}

NEGATIVE_PHRASES = {
    "not good", "very bad", "needs improvement", "worst experience",
    "doesn't help", "no support", "not useful", "not helpful",
    "not clear", "tests patience", "waste of time", "makes no sense",
    "outdated stuff", "nothing works", "straight up trash",
    "complete waste", "total disaster", "zero value", "pure chaos",
    "so stressful", "so draining", "beyond frustrating",
    "absolutely terrible", "genuinely awful", "not working"
}


def rule_based_sentiment(text: str):
    """
    Rule-based fallback for short/slang/informal texts.
    Handles:
      - Single slang words (lit, banger, trash, mid)
      - Negation phrases (not bad, not good, not helpful)
      - Multi-word casual phrases (waste of time, top notch)
    Returns label string or None (let ML decide for longer texts).
    """
    text_lower = text.lower().strip()
    tokens = text_lower.split()

    # Step 1: Check multi-word negative phrases first (highest priority)
    for phrase in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return "Negative"

    # Step 2: Check multi-word positive phrases
    # But skip "not bad" if it also appears in neutral (handle below)
    for phrase in POSITIVE_PHRASES:
        if phrase in text_lower and phrase not in NEUTRAL_PHRASES:
            return "Positive"

    # Step 3: Check neutral phrases
    for phrase in NEUTRAL_PHRASES:
        if phrase in text_lower:
            return "Neutral"

    # Step 4: Single word lookup — only for short texts (1-5 words)
    if len(tokens) <= 5:
        pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        neu_count = sum(1 for t in tokens if t in NEUTRAL_WORDS)

        # If it is a single neutral word like "meh", "okay", "fine" — Neutral
        if len(tokens) == 1 and neu_count > 0:
            return "Neutral"

        # Clear winner
        if pos_count > neg_count and pos_count > neu_count:
            return "Positive"
        elif neg_count > pos_count and neg_count > neu_count:
            return "Negative"
        elif neu_count > 0 and pos_count == 0 and neg_count == 0:
            return "Neutral"

    return None  # Let ML model decide


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Predict sentiment for a single text using trained model.

    Parameters:
        text (str): Raw feedback text.
        model: Trained LogisticRegression model.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        dict: {label, confidence, probabilities}
    """
    # Preprocess the input text
    cleaned = preprocess(text)
    if not cleaned:
        return {"label": "Neutral", "confidence": 0.0, "probabilities": {}}

    # Step 1: Check for sarcasm / contrast patterns first
    is_sarcastic = detect_sarcasm_or_contrast(text)

    # Step 2: Always run ML model
    features = vectorizer.transform([cleaned])
    ml_label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}
    ml_confidence = round(float(max(proba)), 4)

    # Step 3: If sarcasm detected AND model said Positive/Neutral with low confidence
    # flip it to Negative (sarcasm in student feedback almost always = negative)
    if is_sarcastic and ml_label in ("Positive", "Neutral") and ml_confidence < 0.70:
        label = "Negative"
        confidence = round(min(ml_confidence + 0.15, 0.88), 4)  # bump confidence slightly
    else:
        # Step 4: Try rule-based for very short simple texts
        rule_result = rule_based_sentiment(text)
        if rule_result and ml_confidence < 0.55:
            label = rule_result
            confidence = 0.82
        else:
            label = ml_label
            confidence = ml_confidence

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities
    }


def split_into_clauses(text: str) -> list:
    """
    Split feedback into clauses on contrast connectors and punctuation.
    e.g. "Faculty is great but labs are terrible" -> two clauses
    """
    import re
    # Split on but/however/although/though/yet/,/; while keeping content
    parts = re.split(r'[,;]|but|however|although|though|yet|despite', text, flags=re.I)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]


def analyze_feedback(text: str, model, vectorizer) -> dict:
    """
    Full ABSA pipeline for a single feedback string.
    Splits on contrast words so mixed sentences get per-aspect sentiment.
    e.g. "Faculty is great but labs are terrible" ->
         Faculty=Positive, Infrastructure=Negative

    Parameters:
        text (str): Raw student feedback.
        model: Trained model.
        vectorizer: Fitted vectorizer.

    Returns:
        dict: Complete analysis result.
    """
    # Step 1: Split into clauses for per-aspect accuracy
    clauses = split_into_clauses(text)
    if not clauses:
        clauses = [text]

    # Step 2: Map aspects to their best matching clause
    all_aspects = extract_aspects(text)
    aspect_clause_map = {}

    for aspect in all_aspects:
        best_clause = text  # default to full text
        from aspect_extraction import ASPECT_KEYWORDS
        import re
        keywords = ASPECT_KEYWORDS.get(aspect, [])
        for clause in clauses:
            for kw in keywords:
                if re.search(r'' + re.escape(kw) + r'', clause, re.I):
                    best_clause = clause
                    break
        aspect_clause_map[aspect] = best_clause

    # Step 3: Predict sentiment per aspect using its matching clause
    aspect_results = []
    for aspect in all_aspects:
        clause = aspect_clause_map[aspect]
        sr = predict_sentiment(clause, model, vectorizer)
        score = SENTIMENT_SCORES.get(sr["label"], 0.0)
        aspect_results.append({
            "aspect": aspect,
            "sentiment": sr["label"],
            "confidence": sr["confidence"],
            "score": score,
            "color": SENTIMENT_COLORS[sr["label"]],
            "emoji": SENTIMENT_EMOJI[sr["label"]],
            "probabilities": sr["probabilities"]
        })

    # Step 4: Overall sentiment = from full text prediction
    overall_sr = predict_sentiment(text, model, vectorizer)

    # Step 5: Compute overall score as mean of aspect scores
    overall_score = np.mean([r["score"] for r in aspect_results])

    return {
        "original_text": text,
        "processed_text": preprocess(text),
        "aspects": all_aspects,
        "sentiment": overall_sr["label"],
        "confidence": overall_sr["confidence"],
        "overall_score": round(float(overall_score), 3),
        "probabilities": overall_sr["probabilities"],
        "aspect_results": aspect_results,
        "color": SENTIMENT_COLORS[overall_sr["label"]],
        "emoji": SENTIMENT_EMOJI[overall_sr["label"]]
    }


def analyze_batch(texts: list, model, vectorizer) -> list:
    """
    Run ABSA pipeline on a list of feedback texts.

    Parameters:
        texts (list): List of raw feedback strings.

    Returns:
        list: List of analysis result dicts.
    """
    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            result = analyze_feedback(text, model, vectorizer)
            results.append(result)
    return results


# ── Aspect display labels with icons ─────────────────────────────────────────
ASPECT_ICONS = {
    "Faculty":        "🧑‍🏫",
    "Infrastructure": "🏛️",
    "Curriculum":     "📚",
    "Placements":     "💼",
    "Management":     "🏢",
    "General":        "💬",
}

def get_feedback_category(text: str, aspects: list) -> dict:
    """
    Determine what category/topic the feedback belongs to.

    Rules:
    - If text is 1-3 words (single word like 'good', 'lit', 'trash') → Generic
    - If only 'General' aspect detected → Generic
    - Otherwise → return the detected aspect(s) with icons

    Returns:
        dict: {
            'label': display label string,
            'is_generic': bool,
            'aspects': list of aspects with icons,
            'description': short human readable explanation
        }
    """
    tokens = text.strip().split()
    is_short = len(tokens) <= 3
    is_general = aspects == ["General"] or aspects == []

    if is_short or is_general:
        return {
            "label": "Generic Feedback",
            "is_generic": True,
            "aspects": [],
            "description": "This is a short or general expression not tied to a specific aspect of the college."
        }

    aspect_with_icons = [
        {"name": a, "icon": ASPECT_ICONS.get(a, "📌")}
        for a in aspects if a != "General"
    ]

    if len(aspect_with_icons) == 1:
        a = aspect_with_icons[0]
        desc = f"This feedback is specifically about the {a['name']} aspect of the college."
    else:
        names = ", ".join(a["name"] for a in aspect_with_icons)
        desc = f"This feedback covers multiple aspects: {names}."

    return {
        "label": ", ".join(a["name"] for a in aspect_with_icons),
        "is_generic": False,
        "aspects": aspect_with_icons,
        "description": desc
    }


def compute_summary_stats(results: list) -> dict:
    """
    Compute aggregate statistics from a list of analysis results.

    Parameters:
        results (list): List of result dicts from analyze_feedback.

    Returns:
        dict: Summary statistics.
    """
    if not results:
        return {}

    sentiments = [r["sentiment"] for r in results]
    scores = [r["overall_score"] for r in results]

    # Sentiment counts
    pos_count = sentiments.count("Positive")
    neu_count = sentiments.count("Neutral")
    neg_count = sentiments.count("Negative")
    total = len(sentiments)

    # Aspect frequency
    all_aspects = []
    for r in results:
        all_aspects.extend(r["aspects"])

    aspect_counts = {}
    for a in all_aspects:
        aspect_counts[a] = aspect_counts.get(a, 0) + 1

    # Per-aspect sentiment breakdown
    aspect_sentiment = {}
    for r in results:
        for aspect in r["aspects"]:
            if aspect not in aspect_sentiment:
                aspect_sentiment[aspect] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            aspect_sentiment[aspect][r["sentiment"]] += 1

    return {
        "total": total,
        "positive": pos_count,
        "neutral": neu_count,
        "negative": neg_count,
        "positive_pct": round(pos_count / total * 100, 1) if total else 0,
        "neutral_pct": round(neu_count / total * 100, 1) if total else 0,
        "negative_pct": round(neg_count / total * 100, 1) if total else 0,
        "avg_score": round(float(np.mean(scores)), 3) if scores else 0,
        "aspect_counts": aspect_counts,
        "aspect_sentiment": aspect_sentiment,
        "all_texts": ' '.join([r["original_text"] for r in results])
    }
