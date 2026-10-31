
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("disease_model.pkl")
features = joblib.load("features.pkl")

# SQLite database
DB_NAME = "predictions.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            radius REAL,
            texture REAL,
            perimeter REAL,
            area REAL,
            smoothness REAL,
            prediction TEXT,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

create_database()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input"}), 400

        values = []

        for feature in features:
            value = float(data[feature])

            if not np.isfinite(value) or value < 0:
                return jsonify({
                    "error": "Please enter valid non-negative numbers."
                }), 400

            values.append(value)

        input_data = pd.DataFrame(
            [values],
            columns=features
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        # Class 0 = malignant, class 1 = benign
        result = "Benign" if prediction == 1 else "Malignant"

        probability = float(probabilities[prediction])

        # Save prediction history
        conn = sqlite3.connect(DB_NAME)

        conn.execute("""
            INSERT INTO predictions
            (radius, texture, perimeter, area,
             smoothness, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            result,
            probability
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "prediction": result,
            "model_score": round(probability * 100, 2),
            "message": "Educational model output only. Not a medical diagnosis."
        })

    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Please provide all five valid inputs."}), 400

    except Exception:
        return jsonify({"error": "Prediction could not be completed."}), 500


@app.route("/api/history")
def history():

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row

    records = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT 20
    """).fetchall()

    conn.close()

    return jsonify([dict(row) for row in records])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
