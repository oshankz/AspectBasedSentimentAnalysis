"""Utilities for preprocessing student feedback text."""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLP resources once during module import.
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Apply a full NLP preprocessing pipeline to raw text.

    Steps:
    1) Lowercasing
    2) Remove punctuation and non-alphanumeric characters
    3) Tokenization
    4) Stopword removal
    5) Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower().strip()

    # Keep letters, numbers, and whitespace only.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize
    tokens: List[str] = word_tokenize(text)

    # Remove stopwords and short junk tokens, then lemmatize.
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and token not in string.punctuation and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def preprocess_series(series):
    """Preprocess every row in a pandas Series of text values."""
    return series.fillna("").astype(str).apply(preprocess_text)
