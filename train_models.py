import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to files, works in VS Code without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from config import (
    SAMPLE_SIZE, RANDOM_STATE, TEST_SIZE, MAX_BALANCED_PER_CLASS, MODEL_DIR, BASE_DIR,
)
from data_loader import (
    load_dataset, add_binary_label, explore, check_known_domains, make_balanced_dataset,
)
from features import (
    extract_features, feature_names, extract_better_features, better_feature_names,
)


def evaluate(name, y_true, y_pred, save_cm=None):
    print(f"\n===== {name}: MODEL PERFORMANCE =====")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_true, y_pred) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_true, y_pred) * 100:.2f}%")
    print(f"F1 Score  : {f1_score(y_true, y_pred) * 100:.2f}%")
    print(f"\n===== {name}: CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=["Safe", "Malicious"]))

    if save_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Safe", "Malicious"], yticklabels=["Safe", "Malicious"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Malicious URL Detection - {name}")
        out = os.path.join(BASE_DIR, save_cm)
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("Confusion matrix saved to:", out)


def main():
    # 1. Load + explore
    df = load_dataset()
    df = add_binary_label(df)
    explore(df)
    print(df[["url", "type", "label"]].head(10))

    # 2. Sample
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).copy()
    print("Sample size:", len(sample_df))

    # 3. Model 1: Decision Tree, basic features
    X = pd.DataFrame(sample_df["url"].apply(extract_features).tolist(), columns=feature_names)
    y = sample_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Training data:", X_train.shape, "Testing data:", X_test.shape)

    model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=15, min_samples_split=15)
    model.fit(X_train, y_train)
    print("\n✅ Model training completed!")
    y_pred = model.predict(X_test)
    evaluate("Decision Tree (basic features)", y_test, y_pred, save_cm="cm_basic_tree.png")

    print("Training labels:\n", y_train.value_counts())
    print("\nTesting labels:\n", y_test.value_counts())
    print("\nModel prediction distribution:\n", pd.Series(y_pred).value_counts())

    # 4. Model 2: Decision Tree, improved features
    X_better = pd.DataFrame(sample_df["url"].apply(extract_better_features).tolist(),
                            columns=better_feature_names)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_better, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    better_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=20, min_samples_split=10)
    better_model.fit(Xb_train, yb_train)
    print("\n✅ Improved model training completed!")
    evaluate("Decision Tree (improved features)", yb_test, better_model.predict(Xb_test),
             save_cm="cm_better_tree.png")

    # 5. Model 3: TF-IDF + Logistic Regression on the 100k sample
    url_text = sample_df["url"].astype(str)
    Xu_train, Xu_test, yu_train, yu_test = train_test_split(
        url_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000)
    Xu_train_t = vectorizer.fit_transform(Xu_train)
    Xu_test_t = vectorizer.transform(Xu_test)
    url_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    url_model.fit(Xu_train_t, yu_train)
    print("\n✅ URL text model trained!")
    evaluate("TF-IDF + Logistic Regression", yu_test, url_model.predict(Xu_test_t),
             save_cm="cm_tfidf_logreg.png")

    # 6. Check known domains + balanced dataset
    check_known_domains(df)
    balanced_df = make_balanced_dataset(df, MAX_BALANCED_PER_CLASS, RANDOM_STATE)

    # 7. Final model (used by the app): TF-IDF + Logistic Regression on balanced data
    X_tr, X_te, y_tr, y_te = train_test_split(
        balanced_df["url"].astype(str), balanced_df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=balanced_df["label"])
    final_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000)
    X_tr_t = final_vectorizer.fit_transform(X_tr)
    X_te_t = final_vectorizer.transform(X_te)
    final_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    final_model.fit(X_tr_t, y_tr)
    print("\n✅ Final balanced model trained!")
    evaluate("FINAL (balanced TF-IDF + LogReg)", y_te, final_model.predict(X_te_t),
             save_cm="cm_final.png")

    # 8. Save for the app
    joblib.dump(final_vectorizer, os.path.join(MODEL_DIR, "final_vectorizer.joblib"))
    joblib.dump(final_model, os.path.join(MODEL_DIR, "final_model.joblib"))
    print("\n💾 Models saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
