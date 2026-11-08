import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATASET_FILE = BASE_DIR / "medicheck_cleaned.csv"


def main():
    if not DATASET_FILE.exists():
        print("Cleaned dataset not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(DATASET_FILE).fillna("")

    # Combine medicine-related text for search
    text_columns = [
        "name",
        "clean_name",
        "brand_name",
        "generic_name",
        "active_ingredients",
        "dosage_form",
        "standard_form",
        "strength",
        "standard_strength",
    ]

    df["search_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

    # Convert text into numerical TF-IDF vectors
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(df["search_text"])
        # Save the trained search model and its data
    joblib.dump(vectorizer, BASE_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, BASE_DIR / "tfidf_matrix.joblib")
    df.to_pickle(BASE_DIR / "medicine_index.pkl")

    print("Model files saved successfully.")

    print("--- MediCheck ML Model ---")
    print("Medicine records indexed:", len(df))
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    # Test the model with a sample query
    query = "Augmentin 625"
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:5]

    print(f"\nTop results for '{query}':")
    for index in top_indices:
        print(
            f"- {df.iloc[index]['name']} "
            f"(similarity: {scores[index]:.3f})"
        )


if __name__ == "__main__":
    main()
