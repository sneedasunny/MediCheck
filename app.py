from pathlib import Path
import re

import joblib
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

from matcher import Matcher

BASE_DIR = Path(__file__).parent

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = Flask(__name__)
CORS(app)

vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load(BASE_DIR / "tfidf_matrix.joblib")
medicines = pd.read_pickle(BASE_DIR / "medicine_index.pkl")


def tfidf_scores(text):
    vector = vectorizer.transform([text])
    return cosine_similarity(vector, tfidf_matrix).flatten()


# Ingredient / brand / strength / form matching (see matcher.py)
matcher = Matcher(medicines, tfidf_scores)


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value):
    value = clean_value(value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def medicine_to_dict(row, score=None, confidence=None, reasons=None):
    result = {
        "id": clean_value(row.get("id", "")),
        "name": clean_value(row.get("name", "")),
        "brand_name": clean_value(row.get("brand_name", "")),
        "generic_name": clean_value(row.get("generic_name", "")),
        "active_ingredients": clean_value(
            row.get("active_ingredients", "")
        ),
        "dosage_form": clean_value(row.get("dosage_form", "")),
        "strength": clean_value(row.get("strength", "")),
        "side_effects": clean_value(row.get("side_effects", "")),
        "warnings": clean_value(row.get("warnings", "")),
        "interactions": clean_value(row.get("interactions", "")),
        "verification_status": clean_value(
            row.get("verification_status", "")
        ),
    }

    if score is not None:
        result["similarity"] = round(float(score), 3)

    if confidence is not None:
        result["confidence"] = confidence
        result["match_reasons"] = reasons or []

    return result


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/search")
def search_medicines():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "error": "Please enter a medicine name or search term."
        }), 400

    ranked, detected = matcher.rank(query, limit=10)

    results = [
        medicine_to_dict(medicines.iloc[i], score, confidence, reasons)
        for i, score, confidence, reasons in ranked
    ]

    top_confidence = ranked[0][2] if ranked else None

    if top_confidence == "likely":
        confidence_message = (
            "Possible matches found. Verify the medicine name, "
            "strength, and active ingredients."
        )
    else:
        confidence_message = (
            "No strong match. Verify the medicine name "
            "and ingredients carefully."
        )

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results,
        "confidence_message": confidence_message
    })


@app.post("/api/ocr")
def scan_medicine_image():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:
        # Verify the uploaded image.
        image = Image.open(uploaded_file.stream)
        image.verify()

        # Reopen the image after verification.
        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream).convert("RGB")

        # Grayscale, enlarge small text, improve contrast, sharpen.
        image = ImageOps.grayscale(image)
        image = image.resize(
            (image.width * 2, image.height * 2),
            Image.Resampling.LANCZOS
        )
        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = image.filter(ImageFilter.SHARPEN)

        # Try two page layouts; strips have scattered text, so --psm 11
        # (sparse text) often finds words that --psm 6 misses.
        seen_lines = set()
        lines = []
        for psm in (6, 11):
            text = pytesseract.image_to_string(
                image, config=f"--psm {psm}"
            )
            for line in text.splitlines():
                line = line.strip()
                if line and line.casefold() not in seen_lines:
                    seen_lines.add(line.casefold())
                    lines.append(line)

        extracted_text = "\n".join(lines)

        if not extracted_text:
            return jsonify({
                "error": "No text detected. Try a clearer image."
            }), 422

        ranked, detected = matcher.rank(
            extracted_text, limit=10, ocr_mode=True
        )

        candidates = [
            medicine_to_dict(medicines.iloc[i], score, confidence, reasons)
            for i, score, confidence, reasons in ranked
            if confidence != "weak"
        ]

        top_confidence = candidates[0]["confidence"] if candidates else None
        tied = (
            len(candidates) > 1
            and candidates[0]["similarity"] == candidates[1]["similarity"]
        )

        if not candidates:
            note = (
                "Could not identify this medicine from the scan. "
                "Check the printed packaging and try a closer, "
                "glare-free photo of the brand name and strength."
            )
        elif top_confidence == "likely" and not tied:
            note = (
                "Best text match below. This is not a confirmed "
                "identification; check the printed name, strength, "
                "and active ingredients."
            )
        else:
            note = (
                "Only the ingredients could be read, so several "
                "products fit. Choose the one whose brand name and "
                "strength match your packaging."
            )

        return jsonify({
            "extracted_text": extracted_text,
            "detected_ingredients": detected,
            "candidates": candidates,
            "message": note
        })

    except Exception:
        app.logger.exception("OCR image processing failed")

        return jsonify({
            "error": (
                "Could not read this image. Please try another image."
            )
        }), 400


if __name__ == "__main__":
    app.run(debug=True)