import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "malicious_phish.csv")

SAMPLE_SIZE = 100000
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_BALANCED_PER_CLASS = 200000

# Clearly legitimate domains for demonstration
TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "youtube.com",
    "amazon.com",
    "wikipedia.org",
    "apple.com",
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
