# 🎓 Aspect-Based Sentiment Analysis of Student Feedback

A complete end-to-end NLP system that analyses student feedback by identifying **which aspect** (Faculty, Infrastructure, Curriculum, Placements, Management) is being discussed and classifying the **sentiment** (Positive / Neutral / Negative) for each aspect.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🏗️ Project Structure

```
absa_project/
├── app.py                  # Streamlit frontend (run this)
├── data_preprocessing.py   # NLP pipeline (lowercase, tokenize, lemmatize...)
├── aspect_extraction.py    # Keyword-based aspect detection
├── model_training.py       # TF-IDF + Logistic Regression training
├── prediction.py           # End-to-end inference pipeline
├── requirements.txt        # Python dependencies
└── model/                  # Auto-created on first run
    ├── sentiment_model.pkl
    └── tfidf_vectorizer.pkl
```

---

## ⚙️ How It Works

### Pipeline Overview
```
Raw Feedback Text
       ↓
NLP Preprocessing (lowercase → remove punct → tokenize → stopwords → lemmatize)
       ↓
Aspect Extraction (keyword matching → Faculty / Infrastructure / Curriculum / Placements / Management)
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression → Sentiment: Positive / Neutral / Negative
       ↓
Output: Aspect + Sentiment + Score + Confidence
```

### NLP Preprocessing (`data_preprocessing.py`)
- Lowercasing
- Punctuation & special character removal
- Tokenization (NLTK `word_tokenize`)
- Stopword removal (retains negation words like "not", "never")
- Lemmatization (WordNet lemmatizer)

### Aspect Extraction (`aspect_extraction.py`)
Uses curated keyword dictionaries for 5 aspects:
| Aspect | Sample Keywords |
|--------|----------------|
| Faculty | teacher, professor, lecturer, explain, teaching |
| Infrastructure | lab, library, campus, equipment, hostel, wifi |
| Curriculum | syllabus, course, subject, assignment, exam |
| Placements | job, recruit, company, salary, campus drive |
| Management | admin, principal, policy, fee, grievance |

### Model Training (`model_training.py`)
- **Vectorizer**: TF-IDF (5000 features, unigrams + bigrams, log normalization)
- **Classifier**: Logistic Regression (multinomial, balanced class weights)
- **Dataset**: 90 hand-crafted training examples (30 per class)
- **Model persistence**: Saved with `pickle` to `model/` directory

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| Single Text Analysis | Enter feedback → get aspect + sentiment + confidence |
| Batch CSV Upload | Upload a CSV with `feedback` column → bulk analysis |
| Sentiment Cards | Color-coded per-aspect sentiment cards |
| Probability Chart | Bar chart showing model confidence per class |
| Pie Chart | Overall sentiment distribution |
| Bar Chart | Aspect-wise sentiment breakdown |
| Word Cloud | Visual word frequency map |
| Dashboard | Model metrics + confusion matrix |

---

## 📁 CSV Format

Your CSV file must contain at least a `feedback` column:

```csv
feedback
"The faculty is excellent and always available for help."
"Lab equipment is outdated and wifi is slow."
"Placements are great, top companies visit every year."
```

---

## 🎨 UI Color Theme
- 🟢 **Green** = Positive sentiment
- 🟡 **Yellow** = Neutral sentiment
- 🔴 **Red** = Negative sentiment

---

## 🔧 Train Model Manually

```bash
cd absa_project
python model_training.py
```

This trains and saves the model to `model/sentiment_model.pkl` and `model/tfidf_vectorizer.pkl`.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web frontend |
| `scikit-learn` | TF-IDF + Logistic Regression |
| `nltk` | Tokenization, lemmatization, stopwords |
| `pandas` | CSV handling |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `matplotlib` | Word cloud rendering |
| `wordcloud` | Word cloud generation |

---

## 💡 Example Output

**Input:** *"The professors are amazing but infrastructure needs improvement."*

| Aspect | Sentiment | Confidence | Score |
|--------|-----------|------------|-------|
| Faculty | 😊 Positive | 91% | +1.0 |
| Infrastructure | 😞 Negative | 78% | -1.0 |
