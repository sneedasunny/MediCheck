
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "MediCheck_Member3_Search_Database.csv"
OUTPUT_FILE = BASE_DIR / "medicheck_cleaned.csv"


def main():
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Remove columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Normalize text fields without changing missing values
    text_columns = df.select_dtypes(include=["object"]).columns
    for column in text_columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Report missing strength values; do not invent replacements
    print("Missing strength values:", df["strength"].isna().sum())

    # Save the cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Cleaning Complete ---")
    print("Records:", len(df))
    print("Columns:", len(df.columns))
    print("Saved to:", OUTPUT_FILE)
    print("\nRemaining missing values:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
