# PhishGuard AI - Backend Detection Engine

An advanced machine learning inference API built with **Python** and **FastAPI** to identify phishing URLs in real-time. This service serves as the analytical core of the PhishGuard AI ecosystem, communicating with the mobile frontend to provide deep security insights.

## 🚀 Key Features

* **Hybrid Machine Learning Inference:** Utilises a combination of lexical URL features and TF-IDF character n-grams for high-accuracy detection.
* **Real-time URL Normalisation:** Automatically strips mobile-specific tracking parameters and fragments to prevent feature distortion and ensure consistency with training data.
* **Explainable AI (XAI) Integration:** Generates detailed "Analyst Notes" for every scan, providing human-readable justifications for the model's decisions.
* **Probability-Based Classification:** Implements a three-tier sorting system (Safe, Suspicious, Phishing) based on model confidence scores rather than simple binary labels.
* **Asynchronous Processing:** Built on FastAPI to handle concurrent requests from mobile clients with minimal latency.

## 🏗️ System Architecture

The backend follows a modular pipeline design to ensure data integrity from ingestion to prediction:

1.  **Sanitisation Layer:** Validates and cleanses incoming URI strings.
2.  **Feature Extraction Engine:** Calculates 15 distinct structural metrics including Shannon Entropy, subdomain depth, and keyword frequency.
3.  **Vectorisation & Scaling:** Transforms raw data using persistent `TfidfVectorizer` and `StandardScaler` artifacts to maintain the exact environment of the training phase.
4.  **Inference Layer:** Executes a Logistic Regression model on the hybrid feature matrix.

## 🛠️ Technical Stack

* **Framework:** FastAPI (Python 3.13+)
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas & NumPy
* **Serialisation:** Joblib
* **Web Server:** Uvicorn

## 📋 API Specification

### Analyse URL
**Endpoint:** `POST /predict`

**Request Payload:**
```json
{
  "url": "[https://secure-login-update.com/account](https://secure-login-update.com/account)"
}

## Response Payload

```json
{
  "prediction": "Phishing",
  "probability": 0.925,
  "analyst_notes": "The classifier flagged this URL as Phishing with 92.5% confidence. Key indicators: elevated character entropy, sensitive brand-related keywords identified.",
  "url": "https://secure-login-update.com/account"
}
```

---

## 🧑‍💻 Academic Context

This backend was developed as the primary data source for an Advanced Mobile Application Development module. It demonstrates proficiency in:

- **Model Persistence:** Saving and loading complex ML pipelines for production use.
- **Feature Engineering:** Designing robust lexical extraction logic for security-sensitive data.
- **Handling Distribution Shift:** Mitigating the gap between static datasets and real-world mobile URL traffic.

---

## ⚙️ Setup Instructions

### Environment Setup

```bash
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

---
