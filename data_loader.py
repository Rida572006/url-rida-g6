import os
import zipfile
import pandas as pd

from config import CSV_PATH, DATA_DIR


def load_dataset():
    """Load malicious_phish.csv from data/. If only a .zip is there, extract it first."""
    if not os.path.exists(CSV_PATH):
        zips = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".zip")]
        if not zips:
            raise FileNotFoundError(
                f"Dataset not found. Put malicious_phish.csv (or the dataset .zip) in: {DATA_DIR}"
            )
        with zipfile.ZipFile(os.path.join(DATA_DIR, zips[0]), "r") as z:
            z.extractall(DATA_DIR)
        print("Extracted files:", os.listdir(DATA_DIR))

    df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    return df


def add_binary_label(df):
    """benign -> 0, everything else -> 1"""
    df["label"] = df["type"].apply(lambda x: 0 if x == "benign" else 1)
    return df


def explore(df):
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nURL types:")
    print(df["type"].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nLabel distribution:")
    print(df["label"].value_counts())


def check_known_domains(df):
    domains = ["google.com", "microsoft.com", "youtube.com",
               "amazon.com", "wikipedia.org", "apple.com"]
    for domain in domains:
        matches = df[df["url"].astype(str).str.contains(domain, case=False, na=False)]
        print("\n" + "=" * 60)
        print(domain)
        print("Total matches:", len(matches))
        if len(matches) > 0:
            print("\nClass distribution:")
            print(matches["type"].value_counts())
            print("\nExamples:")
            print(matches[["url", "type"]].head(3).to_string(index=False))


def make_balanced_dataset(df, max_per_class, random_state):
    benign_data = df[df["type"] == "benign"]
    malicious_data = df[df["type"] != "benign"]
    print("Benign URLs:", len(benign_data))
    print("Malicious URLs:", len(malicious_data))

    n = min(len(benign_data), len(malicious_data), max_per_class)
    balanced_df = pd.concat(
        [benign_data.sample(n=n, random_state=random_state),
         malicious_data.sample(n=n, random_state=random_state)],
        ignore_index=True,
    )
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced_df["label"] = balanced_df["type"].apply(lambda x: 0 if x == "benign" else 1)

    print("\nBalanced dataset shape:", balanced_df.shape)
    print("\nLabel distribution:")
    print(balanced_df["label"].value_counts())
    return balanced_df
