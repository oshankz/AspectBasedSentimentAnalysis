"""
data_preprocessing.py
---------------------
Handles all NLP preprocessing steps for student feedback text.
Includes: lowercasing, punctuation removal, stopword removal,
tokenization, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (run once)
def download_nltk_resources():
    """Download all required NLTK datasets."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# Keep negation words — they affect sentiment
NEGATION_WORDS = {'no', 'not', 'never', 'neither', 'nor', 'hardly', 'barely', 'scarcely'}
STOP_WORDS -= NEGATION_WORDS


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters from text."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_special_characters(text: str) -> str:
    """Remove numbers and special characters, keep only alphabetic words."""
    return re.sub(r'[^a-zA-Z\s]', '', text)


def tokenize(text: str) -> list:
    """Split text into individual word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove common stopwords while retaining negation words."""
    return [token for token in tokens if token not in STOP_WORDS]


def lemmatize_tokens(tokens: list) -> list:
    """Reduce each token to its base/lemma form."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline.
    Returns a single cleaned string ready for feature extraction.

    Steps:
    1. Lowercase
    2. Remove punctuation & special chars
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    return ' '.join(tokens)


def preprocess_batch(texts: list) -> list:
    """Apply preprocessing to a list of feedback texts."""
    return [preprocess(t) for t in texts]
