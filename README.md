# Aspect Based Sentiment Analysis of Student Feedback

A complete end-to-end NLP project built with **Python**, **Logistic Regression**, and **Streamlit**.

## Features
- Input feedback from:
  - Single text box
  - CSV upload
- NLP preprocessing:
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Tokenization
  - Lemmatization
- Aspect identification via keyword mapping for:
  - Faculty
  - Infrastructure
  - Curriculum
  - Placements
  - Management
- Sentiment classification using:
  - TF-IDF
  - Logistic Regression
- Dashboard outputs:
  - Aspect-wise sentiment cards
  - Bar and pie charts
  - Word cloud
  - Summary metrics

## Project Structure
- `data_preprocessing.py`
- `aspect_extraction.py`
- `model_training.py`
- `prediction.py`
- `app.py`
- `requirements.txt`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Notes
- Model artifact is saved as `sentiment_model.pkl` after training.
- If no training CSV is supplied, the app uses a built-in sample dataset for quick start.
