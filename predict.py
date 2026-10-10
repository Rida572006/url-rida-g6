import os
import joblib
from urllib.parse import urlparse

from config import MODEL_DIR, TRUSTED_DOMAINS

_vec_path = os.path.join(MODEL_DIR, "final_vectorizer.joblib")
_model_path = os.path.join(MODEL_DIR, "final_model.joblib")

if not (os.path.exists(_vec_path) and os.path.exists(_model_path)):
    raise FileNotFoundError("Trained models not found. Run `python train_models.py` first.")

final_vectorizer = joblib.load(_vec_path)
final_model = joblib.load(_model_path)


def hybrid_predict_url(url):
    url = (url or "").strip()

    if not url:
        return "⚠️ Please enter a URL."

    # Add scheme if user doesn't provide one
    check_url = url if "://" in url else "https://" + url

    try:
        hostname = (urlparse(check_url).hostname or "").lower()
    except ValueError:
        hostname = ""

    # Remove www.
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Trusted-domain check
    if hostname in TRUSTED_DOMAINS:
        return "🟢 SAFE URL\nVerified trusted domain"

    # Otherwise use ML model
    url_vector = final_vectorizer.transform([url])
    prediction = final_model.predict(url_vector)[0]
    probabilities = final_model.predict_proba(url_vector)[0]
    confidence = max(probabilities) * 100

    if prediction == 1:
        return f"🔴 MALICIOUS / PHISHING URL\nML Confidence: {confidence:.2f}%"
    return f"🟢 SAFE URL\nML Confidence: {confidence:.2f}%"


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://www.youtube.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.apple.com",
    ]
    for u in test_urls:
        print(u)
        print(hybrid_predict_url(u))
        print("-" * 50)
