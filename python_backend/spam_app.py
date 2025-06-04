from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import load_model
from lime.lime_text import LimeTextExplainer
import pickle
import numpy as np
import os
import sys
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dari modul lokal
from python_backend.utils.spam_preprocess_text import (
    cleaning, casefolding, handle_slangwords,
    tokenizing, remove_stopwords, text_result, preprocess_text
)
from python_backend.utils.spam_predictor import predictor
from python_backend.utils.spam_rule_based_filter import rule_based_spam_filter
from python_backend.utils.spam_predict_and_explain import predict_and_explain_spam

# Inisialisasi FastAPI
app = FastAPI(
    title="Spam Detection API",
    description="API untuk mendeteksi apakah pesan atau teks yang diberikan merupakan spam atau tidak.",
)

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua metode HTTP (GET, POST, dll.)
    allow_headers=["*"],  # Mengizinkan semua headers
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_spam.h5")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# Load model & tools
model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

explainer = LimeTextExplainer(class_names=['Not SPAM', 'SPAM'])

# Define input schema
class TextInput(BaseModel):
    text: str

@app.post("/predict", tags=["Spam Detection"], summary="Deteksi apakah pesan merupakan spam atau tidak")
async def predict(input_data: TextInput):
    text = input_data.text
    pred_class, prob, explanation, source = predict_and_explain_spam(
        text=text,
        model=model,
        vectorizer=vectorizer,
        explainer=explainer,
        class_names=['Not SPAM', 'SPAM']
    )
    return {
        "prediction": pred_class,
        "probability": round(prob, 4),
        "explanation": explanation,
        "source": source
    }
