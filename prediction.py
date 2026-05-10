"""
prediction.py
-------------
End-to-end prediction for Aspect-Based Sentiment Analysis.
Significantly improved with:
  - Proper negation handling (not good, never helpful, barely works)
  - Intensifier boosting (very bad, extremely poor, absolutely useless)
  - Contrastive clause splitting (but, however, although...)
  - Sarcasm / dark-humour detection
  - Multi-word phrase dictionary
  - Confidence-gated rule override
"""

import os
import re
import numpy as np
from data_preprocessing import preprocess
from aspect_extraction import extract_aspects, ASPECT_KEYWORDS
from model_training import load_model, train_model

MODEL_PATH      = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

SENTIMENT_SCORES = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
SENTIMENT_EMOJI  = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}


# ═══════════════════════════════════════════════════════════════════════
# WORD LISTS
# ═══════════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","superb",
    "outstanding","brilliant","awesome","best","love","loved","happy",
    "perfect","beautiful","clean","modern","helpful","supportive",
    "knowledgeable","talented","efficient","effective","impressive","top",
    "well","better","positive","recommend","enjoy","enjoyed","useful",
    "strong","smart","innovative","quality","satisfied","satisfying",
    "rewarding","exceptional","remarkable","splendid","nice","decent",
    "lit","banger","fire","solid","dope","clutch","smooth","nailed",
    "killed","goated","blessed","legendary","phenomenal","stellar",
    "mint","crisp","ace","top-notch","first-rate","commendable","superb",
    "praiseworthy","noteworthy","admirable","delightful","pleasing",
    "enriching","motivating","inspiring","passionate","dedicated","caring",
    "approachable","available","responsive","transparent","fair","organised",
    "updated","relevant","practical","comprehensive","structured","balanced",
    "industry-ready","employable","skilled","active","proactive"
}

NEGATIVE_WORDS = {
    "bad","poor","terrible","awful","horrible","worst","useless","pathetic",
    "dirty","broken","outdated","slow","boring","rude","arrogant","corrupt",
    "rigid","unhelpful","disappointing","disappointed","frustrating",
    "frustrated","waste","hate","hated","ugly","disgusting","unhygienic",
    "overcrowded","disorganized","irresponsible","careless","negligent",
    "incompetent","inadequate","trash","weak","buggy","messy","painful",
    "exhausting","stressful","draining","confusing","chaotic","pointless",
    "unbearable","ridiculous","nightmare","sucks","atrocious","dreadful",
    "abysmal","diabolical","dismal","outdated","irrelevant","repetitive",
    "unavailable","unresponsive","inactive","disorganised","bureaucratic",
    "unfair","biased","rushed","boring","demotivating","hopeless",
    "crumbling","dilapidated","non-functional","overcrowded","understaffed",
    "underfunded","mismanaged","overpriced","insufficient","lacking",
    "neglected","ignored","delayed","cancelled","broken","non-existent"
}

NEUTRAL_WORDS = {
    "okay","fine","average","manageable","fair","acceptable","normal",
    "standard","typical","alright","meh","moderate","mediocre","passable",
    "adequate","ordinary","mid","basic","decent","conventional","regular"
}

# Negation words — flip the sentiment of what follows
NEGATION_WORDS = {
    "not","never","no","neither","nor","hardly","barely","scarcely",
    "doesn't","don't","didn't","isn't","aren't","wasn't","weren't",
    "can't","cannot","won't","wouldn't","shouldn't","couldn't","haven't",
    "hasn't","hadn't","nothing","nowhere","nobody","none","without",
    "lack","lacking","lacks","lacked","absence","absent","fails","failed",
    "fail","refuse","refused","deny","denied","ignore","ignores","ignored"
}

# Intensifiers — amplify the next word's sentiment
INTENSIFIERS = {
    "very","extremely","absolutely","completely","totally","utterly",
    "incredibly","remarkably","exceptionally","highly","deeply","severely",
    "terribly","horribly","awfully","dreadfully","shockingly","genuinely",
    "truly","really","so","such","quite","rather","fairly","pretty",
    "super","ultra","mega","insanely","ridiculously","painfully","dangerously"
}

# Diminishers — weaken sentiment toward neutral
DIMINISHERS = {
    "somewhat","slightly","a bit","a little","kind of","sort of","rather",
    "fairly","partially","mostly","generally","usually","sometimes","often",
    "occasionally","at times","to some extent","in some ways","relatively"
}


# ═══════════════════════════════════════════════════════════════════════
# PHRASE DICTIONARIES  (checked before ML model)
# ═══════════════════════════════════════════════════════════════════════

NEGATIVE_PHRASES = {
    # Negation + positive = negative
    "not good","not great","not helpful","not useful","not clear",
    "not available","not responsive","not updated","not relevant",
    "not working","not satisfied","not impressive","not effective",
    "not worth it","not at all","not even close","not up to the mark",
    "never available","never helps","never responds","never works",
    "barely works","barely functional","barely adequate","barely useful",
    "hardly helpful","hardly ever available","hardly satisfying",
    # Direct negative phrases
    "needs improvement","needs a lot of improvement","needs work",
    "waste of time","waste of money","complete waste","total waste",
    "not worth","poor quality","very poor","extremely poor",
    "worst experience","worst ever","very disappointing","deeply disappointing",
    "beyond frustrating","absolutely terrible","genuinely awful",
    "straight up trash","zero value","pure chaos","total disaster",
    "nothing works","makes no sense","no support","no guidance",
    "no help","no response","no communication","no transparency",
    "doesn't help","don't recommend","would not recommend",
    "very slow","too slow","extremely slow","very late","always late",
    "always absent","always unavailable","never on time",
    "could not understand","hard to understand","impossible to follow",
    "very difficult to approach","not approachable","very rude",
    "very arrogant","very unprofessional","very irresponsible",
    "leaves early","skips class","skips topics","ignores students",
    "reads from slides","no practical","no hands on","outdated content",
    "outdated syllabus","outdated equipment","broken equipment",
    "broken computers","no wifi","very slow wifi","wifi doesn't work",
    "labs don't work","equipment doesn't work","library lacks",
    "not enough books","no books","canteen is dirty","food is bad",
    "food is unhygienic","very expensive","too expensive","very high fees",
    "fees are too high","no value for money","not worth the fees"
}

POSITIVE_PHRASES = {
    "very good","really good","so good","pretty good","quite good",
    "extremely good","absolutely good","very helpful","super helpful",
    "very supportive","always available","always helpful","always ready",
    "very knowledgeable","highly knowledgeable","very experienced",
    "very approachable","easy to approach","very clear","very well explained",
    "explains well","explains clearly","explains very well",
    "highly recommend","would recommend","strongly recommend",
    "top notch","on point","worth it","great stuff","loved it",
    "works well","amazing experience","exceeded expectations",
    "no complaints","actually impressive","surprisingly good",
    "really impressive","very impressive","truly impressive",
    "very modern","up to date","well maintained","well equipped",
    "well structured","well organised","well designed","well planned",
    "very clean","very comfortable","very spacious","very fast",
    "very reliable","very efficient","very effective","very active",
    "very dedicated","very passionate","very caring","very fair",
    "very transparent","very responsive","very proactive",
    "highly dedicated","highly effective","highly efficient",
    "best faculty","best infrastructure","best curriculum",
    "top companies","high salary","great packages","good placements",
    "strong placements","many companies","top mncs"
}

NEUTRAL_PHRASES = {
    "not bad","could be better","decent enough","nothing special",
    "does the job","gets the work done","so so","kind of okay",
    "pretty average","not bad not great","somewhere in the middle",
    "neither good nor bad","okay i guess","it is what it is",
    "nothing to write home about","meets expectations","just okay",
    "acceptable but","passable enough","fairly standard","fairly average",
    "some are good some are bad","mixed experience","mixed bag",
    "has potential but","needs some improvement","could improve",
    "not too bad","not too good","average at best","works sometimes",
    "sometimes good sometimes bad","inconsistent","depends on"
}

# Sarcasm patterns — positive surface, negative meaning
SARCASM_PATTERNS = [
    re.compile(r'\b(great|amazing|excellent|wonderful|fantastic|brilliant|superb|love|perfect)\b.{0,60}\b(when|if|except|unless|only if|after only|just|barely|never|no one|zero|nothing|broken|useless|pathetic|terrible|awful|horrible|waste)', re.I),
    re.compile(r'\b(if your goal is|prepares you for|builds character|rich tradition of|consistently|teaches you patience|teaches you survival)\b', re.I),
    re.compile(r'\b(belong in a museum|from the 90s|from the 1990s|stopped existing|do not exist|never use|never existed)\b', re.I),
    re.compile(r'\b(question my life|life choices|destroying us|break.*spirit|feel.*hopeless|nostalgic for freedom|feel.*hopeless|expert at pretending|pretending to learn)\b', re.I),
    re.compile(r'\b(replied after only|responded after|took a year|took months|after three months|after six months)\b', re.I),
    re.compile(r'\b(counted the rejections|count the failures|measure the disappointments)\b', re.I),
    re.compile(r'\b(only three|only two|only one).{0,30}\b(out of|from|among)\b', re.I),
    re.compile(r'\bworks.{0,20}\b(exactly|only|for just|for about)\b.{0,20}\b(minute|second|hour|day|week)\b', re.I),
    re.compile(r'\b(historically|historically speaking|in the sense that|in a way|technically)\b.{0,40}\b(nothing|no|never|hasn|haven|didn|don|can\'t|cannot)\b', re.I),
]

CONTRAST_RE = re.compile(
    r'\b(but|however|although|though|yet|despite|unfortunately|sadly|except|while|whereas|on the other hand|that said|having said that|even so|in contrast|nevertheless|nonetheless)\b',
    re.I
)


# ═══════════════════════════════════════════════════════════════════════
# CORE NLP HELPERS
# ═══════════════════════════════════════════════════════════════════════

def detect_sarcasm(text: str) -> bool:
    for p in SARCASM_PATTERNS:
        if p.search(text):
            return True
    return False


def detect_negation_context(text: str) -> str:
    """
    Walk through tokens and detect negation + intensifier context.
    Returns adjusted sentiment hint: 'positive', 'negative', 'neutral', or None.
    """
    tokens = re.findall(r"[\w']+", text.lower())
    n = len(tokens)

    neg_score = 0
    pos_score = 0
    negated   = False
    intensify = 1.0

    i = 0
    while i < n:
        tok = tokens[i]

        if tok in NEGATION_WORDS:
            negated  = True
            intensify = 1.0
            i += 1
            continue

        if tok in INTENSIFIERS:
            intensify = 1.5
            i += 1
            continue

        if tok in DIMINISHERS:
            intensify = 0.6
            i += 1
            continue

        if tok in POSITIVE_WORDS:
            weight = intensify
            if negated:
                neg_score += weight * 1.2   # negated positive = stronger negative signal
            else:
                pos_score += weight
            negated   = False
            intensify = 1.0

        elif tok in NEGATIVE_WORDS:
            weight = intensify
            if negated:
                pos_score += weight * 0.7   # negated negative = weak positive signal
            else:
                neg_score += weight * 1.2   # negative words count more
            negated   = False
            intensify = 1.0

        elif tok in NEUTRAL_WORDS:
            negated   = False
            intensify = 1.0

        # Reset negation after 3 tokens of gap
        elif negated:
            pass  # keep negation active for compound phrases

        i += 1

    if neg_score == 0 and pos_score == 0:
        return None

    diff = pos_score - neg_score
    total = pos_score + neg_score

    if diff > 0.4 * total:
        return "positive"
    elif diff < -0.3 * total:
        return "negative"
    elif abs(diff) <= 0.3 * total and total > 0:
        return "neutral"
    return None


def check_phrases(text: str):
    """
    Check multi-word phrase dictionaries.
    Returns 'Positive', 'Negative', 'Neutral', or None.
    Priority: Negative > Positive > Neutral
    """
    t = text.lower()

    # Negative phrases have highest priority
    for phrase in NEGATIVE_PHRASES:
        if phrase in t:
            return "Negative"

    for phrase in POSITIVE_PHRASES:
        if phrase in t and phrase not in NEUTRAL_PHRASES:
            return "Positive"

    for phrase in NEUTRAL_PHRASES:
        if phrase in t:
            return "Neutral"

    return None


def count_sentiment_words(text: str):
    """
    Count positive, negative, neutral words in text.
    Returns (pos, neg, neu) counts.
    """
    tokens = re.findall(r"[\w']+", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    neu = sum(1 for t in tokens if t in NEUTRAL_WORDS)
    return pos, neg, neu


def get_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return load_model(MODEL_PATH, VECTORIZER_PATH)
    else:
        print("No saved model found. Training now...")
        train_model()
        return load_model(MODEL_PATH, VECTORIZER_PATH)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Predict sentiment using a multi-layer approach:
      1. Sarcasm detection (overrides ML if triggered)
      2. Multi-word phrase lookup
      3. ML model prediction
      4. Negation-context analysis
      5. Word-count tiebreaker for short texts
    """
    cleaned = preprocess(text)
    if not cleaned:
        return {"label": "Neutral", "confidence": 0.5, "probabilities": {}}

    text_lower = text.lower().strip()
    tokens     = re.findall(r"[\w']+", text_lower)
    word_count = len(tokens)

    # ── Step 1: ML model (always run) ────────────────────────────────
    features   = vectorizer.transform([cleaned])
    ml_label   = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    classes    = model.classes_
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    ml_conf    = round(float(max(proba)), 4)

    # Sorted probabilities for decision making
    sorted_proba = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    top_label, top_conf     = sorted_proba[0]
    second_label, second_conf = sorted_proba[1] if len(sorted_proba) > 1 else (top_label, 0)

    # ── Step 2: Sarcasm detection ─────────────────────────────────────
    is_sarcastic = detect_sarcasm(text)
    if is_sarcastic and top_label in ("Positive", "Neutral"):
        # Sarcasm almost always means negative in student feedback
        label      = "Negative"
        confidence = round(max(top_conf, 0.72), 4)
        return {"label": label, "confidence": confidence, "probabilities": proba_dict}

    # ── Step 3: Phrase dictionary check ──────────────────────────────
    phrase_result = check_phrases(text)

    # ── Step 4: Negation-context analysis ────────────────────────────
    negation_hint = detect_negation_context(text)

    # ── Step 5: Word counts ───────────────────────────────────────────
    pos_cnt, neg_cnt, neu_cnt = count_sentiment_words(text)

    # ── Step 6: Contrast detection ────────────────────────────────────
    has_contrast = bool(CONTRAST_RE.search(text))

    # ── Decision logic ────────────────────────────────────────────────

    # HIGH confidence ML — trust it unless phrase dict strongly disagrees
    if ml_conf >= 0.75:
        # But check: phrase dict strongly contradicts ML
        if phrase_result and phrase_result != ml_label:
            # Strong phrase signal vs confident ML — use phrase if word evidence agrees
            if phrase_result == "Negative" and neg_cnt > pos_cnt:
                label, confidence = "Negative", round(max(ml_conf - 0.1, 0.65), 4)
            elif phrase_result == "Positive" and pos_cnt > neg_cnt:
                label, confidence = "Positive", round(max(ml_conf - 0.1, 0.65), 4)
            else:
                label, confidence = ml_label, ml_conf
        else:
            label, confidence = ml_label, ml_conf

    # MEDIUM confidence ML (0.55–0.75) — use all signals
    elif 0.55 <= ml_conf < 0.75:
        signals = []

        if phrase_result:
            signals.append(phrase_result)
        if negation_hint:
            signals.append(negation_hint.capitalize())

        # Word count signal
        if pos_cnt > neg_cnt * 1.5:
            signals.append("Positive")
        elif neg_cnt > pos_cnt * 1.2:
            signals.append("Negative")
        elif pos_cnt == neg_cnt and neu_cnt > 0:
            signals.append("Neutral")

        if signals:
            from collections import Counter
            vote = Counter(signals)
            best_signal, _ = vote.most_common(1)[0]

            if best_signal == ml_label:
                # Agreement: boost confidence
                label, confidence = ml_label, round(min(ml_conf + 0.08, 0.90), 4)
            elif vote[best_signal] >= 2:
                # Strong disagreement: override ML
                label, confidence = best_signal, round(ml_conf - 0.05, 4)
            else:
                # Weak disagreement: stay with ML
                label, confidence = ml_label, ml_conf
        else:
            label, confidence = ml_label, ml_conf

    # LOW confidence ML (< 0.55) — rule-based takes over
    else:
        if phrase_result:
            label, confidence = phrase_result, 0.78

        elif negation_hint:
            label, confidence = negation_hint.capitalize(), 0.72

        elif word_count <= 6:
            # Short text: use word counts
            if neg_cnt > pos_cnt:
                label, confidence = "Negative", 0.75
            elif pos_cnt > neg_cnt:
                label, confidence = "Positive", 0.75
            elif neu_cnt > 0:
                label, confidence = "Neutral", 0.70
            else:
                label, confidence = ml_label, ml_conf

        else:
            # Longer ambiguous text — lean on word balance
            if neg_cnt > pos_cnt * 1.3:
                label, confidence = "Negative", 0.68
            elif pos_cnt > neg_cnt * 1.3:
                label, confidence = "Positive", 0.68
            else:
                label, confidence = "Neutral", 0.62

    # ── Final sanity check: contrast sentences ────────────────────────
    # "X is great but Y is terrible" — if both pos and neg words present
    # with contrast word, the negative side dominates in student context
    if has_contrast and pos_cnt > 0 and neg_cnt > 0 and label == "Positive":
        if neg_cnt >= pos_cnt:
            label      = "Neutral" if neg_cnt == pos_cnt else "Negative"
            confidence = round(confidence * 0.85, 4)

    return {
        "label":         label,
        "confidence":    round(float(confidence), 4),
        "probabilities": proba_dict
    }


# ═══════════════════════════════════════════════════════════════════════
# CLAUSE SPLITTING
# ═══════════════════════════════════════════════════════════════════════

def split_into_clauses(text: str) -> list:
    """
    Split feedback into clauses on contrast connectors and punctuation.
    Ensures each clause is meaningful (> 4 words).
    """
    parts = re.split(
        r'[;]|\b(?:but|however|although|though|yet|despite|whereas|while|except|nevertheless|nonetheless|on the other hand|that said|even so)\b',
        text, flags=re.I
    )
    result = []
    for p in parts:
        p = p.strip().strip(',').strip()
        if p and len(p.split()) >= 3:
            result.append(p)
    return result if result else [text]


# ═══════════════════════════════════════════════════════════════════════
# FULL ABSA PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def analyze_feedback(text: str, model, vectorizer) -> dict:
    """
    Full ABSA pipeline:
    1. Split into clauses
    2. Map each detected aspect to its best clause
    3. Predict sentiment per-clause (per-aspect)
    4. Predict overall sentiment on full text
    """
    clauses = split_into_clauses(text)
    all_aspects = extract_aspects(text)

    # Map each aspect to the clause that mentions it
    aspect_clause_map = {}
    for aspect in all_aspects:
        keywords    = ASPECT_KEYWORDS.get(aspect, [])
        best_clause = text  # default: full text
        for clause in clauses:
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', clause, re.I):
                    best_clause = clause
                    break
        aspect_clause_map[aspect] = best_clause

    # Predict sentiment per aspect
    aspect_results = []
    for aspect in all_aspects:
        clause = aspect_clause_map[aspect]
        sr     = predict_sentiment(clause, model, vectorizer)
        score  = SENTIMENT_SCORES.get(sr["label"], 0.0)
        aspect_results.append({
            "aspect":       aspect,
            "sentiment":    sr["label"],
            "confidence":   sr["confidence"],
            "score":        score,
            "color":        SENTIMENT_COLORS[sr["label"]],
            "emoji":        SENTIMENT_EMOJI[sr["label"]],
            "probabilities": sr["probabilities"]
        })

    # Overall sentiment from full text
    overall_sr    = predict_sentiment(text, model, vectorizer)
    overall_score = np.mean([r["score"] for r in aspect_results]) if aspect_results else SENTIMENT_SCORES.get(overall_sr["label"], 0.0)

    return {
        "original_text":  text,
        "processed_text": preprocess(text),
        "aspects":        all_aspects,
        "sentiment":      overall_sr["label"],
        "confidence":     overall_sr["confidence"],
        "overall_score":  round(float(overall_score), 3),
        "probabilities":  overall_sr["probabilities"],
        "aspect_results": aspect_results,
        "color":          SENTIMENT_COLORS[overall_sr["label"]],
        "emoji":          SENTIMENT_EMOJI[overall_sr["label"]]
    }


def analyze_batch(texts: list, model, vectorizer) -> list:
    return [
        analyze_feedback(str(t), model, vectorizer)
        for t in texts
        if isinstance(t, str) and t.strip()
    ]


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════

ASPECT_ICONS = {
    "Faculty":        "🧑‍🏫",
    "Infrastructure": "🏛️",
    "Curriculum":     "📚",
    "Placements":     "💼",
    "Management":     "🏢",
    "General":        "💬",
}


def get_feedback_category(text: str, aspects: list) -> dict:
    tokens     = text.strip().split()
    is_short   = len(tokens) <= 3
    is_general = aspects == ["General"] or aspects == []
    if is_short or is_general:
        return {
            "label": "Generic Feedback", "is_generic": True, "aspects": [],
            "description": "Short or general expression not tied to a specific aspect."
        }
    aspect_with_icons = [{"name": a, "icon": ASPECT_ICONS.get(a, "📌")} for a in aspects if a != "General"]
    names = ", ".join(a["name"] for a in aspect_with_icons)
    desc  = (f"Feedback about the {aspect_with_icons[0]['name']} aspect."
             if len(aspect_with_icons) == 1
             else f"Feedback covers: {names}.")
    return {"label": names, "is_generic": False, "aspects": aspect_with_icons, "description": desc}


def compute_summary_stats(results: list) -> dict:
    if not results:
        return {}
    sentiments = [r["sentiment"] for r in results]
    scores     = [r["overall_score"] for r in results]
    pos = sentiments.count("Positive")
    neu = sentiments.count("Neutral")
    neg = sentiments.count("Negative")
    total = len(sentiments)

    aspect_sentiment = {}
    for r in results:
        for aspect in r["aspects"]:
            if aspect not in aspect_sentiment:
                aspect_sentiment[aspect] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            aspect_sentiment[aspect][r["sentiment"]] += 1

    all_aspects = []
    for r in results:
        all_aspects.extend(r["aspects"])
    aspect_counts = {}
    for a in all_aspects:
        aspect_counts[a] = aspect_counts.get(a, 0) + 1

    return {
        "total":            total,
        "positive":         pos,
        "neutral":          neu,
        "negative":         neg,
        "positive_pct":     round(pos / total * 100, 1) if total else 0,
        "neutral_pct":      round(neu / total * 100, 1) if total else 0,
        "negative_pct":     round(neg / total * 100, 1) if total else 0,
        "avg_score":        round(float(np.mean(scores)), 3) if scores else 0,
        "aspect_counts":    aspect_counts,
        "aspect_sentiment": aspect_sentiment,
        "all_texts":        " ".join(r["original_text"] for r in results)
    }
