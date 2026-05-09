import joblib
import pandas as pd
import numpy as np
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix
from fastapi.middleware.cors import CORSMiddleware

# Import the extraction logic from my feature_engineering.py
from src.feature_engineering import extract_url_features, clean_url

app = FastAPI(title="PhishGuard AI: Advanced Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Global Persistence Layer: Loading Model Artifacts
# ---------------------------------------------------------
MODELS_DIR = "models"

print("Loading AI model and preprocessing artifacts...")
try:
    model = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    print("All artifacts successfully loaded. System is operational.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialise models. {e}")

class URLInput(BaseModel):
    url: str

# ---------------------------------------------------------
# NEW: Real-Time Mobile URL Normalisation
# ---------------------------------------------------------
def normalise_for_inference(url: str) -> str:
    """
    Strips mobile noise (tracking params, fragments) that wasn't in
    the training dataset to prevent feature distortion.
    """
    # Remove whitespace and lowercase
    u = url.strip().lower()
    # Strip URL fragments (#)
    u = u.split('#')[0]
    # Strip common query parameters that skew URL length/entropy
    u = u.split('?')[0]
    return u

def generate_analyst_notes(label, prob, features):
    """
    Generates XAI justification based on confidence and features.
    """
    if label == "Safe" and prob > 0.8:
        return "The URL exhibits structural characteristics consistent with legitimate traffic."

    findings = []
    if features.get('url_entropy', 0) > 4.2:
        findings.append("elevated character entropy (randomness)")
    if features.get('suspicious_keyword_count', 0) > 0:
        findings.append("sensitive brand-related keywords detected")
    if features.get('subdomain_depth', 0) > 2:
        findings.append("excessive subdomain nesting")
    if features.get('is_https', 1) == 0:
        findings.append("unencrypted HTTP protocol")

    report = f"Analysis complete. Confidence: {prob:.1%}."
    if findings:
        report += " Key indicators: " + ", ".join(findings) + "."
    elif label == "Suspicious":
        report += " URL shows ambiguous patterns; exercise caution."
    return report

# ---------------------------------------------------------
# Inference Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict(data: URLInput):
    try:
        # 1. URI Sanitisation & NEW Inference Normalisation
        # Fixed: We use normalisation to match the training data 'style'
        raw_url = clean_url(data.url)
        processed_url = normalise_for_inference(raw_url)

        # 2. Feature Extraction
        lex_features = extract_url_features(processed_url)
        lex_df = pd.DataFrame([lex_features])[feature_names]

        # 3. Parameter Transformation
        lex_scaled = scaler.transform(lex_df)
        tfidf_matrix = tfidf.transform([processed_url])

        # 4. Hybrid Matrix Construction
        hybrid_matrix = hstack([lex_scaled, tfidf_matrix])

        # 5. Inference with Probability Thresholding
        # Instead of just .predict(), we use .predict_proba() for sorting
        probs = model.predict_proba(hybrid_matrix)[0]
        phishing_prob = float(probs[1]) # Probability of class 1 (Phishing)

        # 6. Sorting Logic (Threshold-based)
        # 0.0 - 0.4: Safe | 0.4 - 0.7: Suspicious | 0.7 - 1.0: Phishing
        if phishing_prob >= 0.70:
            label = "Phishing"
        elif phishing_prob >= 0.40:
            label = "Suspicious"
        else:
            label = "Safe"

        # 7. Generate XAI Notes
        notes = generate_analyst_notes(label, phishing_prob, lex_features)

        return {
            "prediction": label,
            "probability": phishing_prob,
            "analyst_notes": notes,
            "url": processed_url,
            "raw_url": raw_url
        }

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
