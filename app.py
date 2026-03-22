from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, pandas as pd, os, sys

# ── Path setup ────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, template_folder="apps/templates")

# ── Load all model artifacts once at startup ──────────────
model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
explainer    = joblib.load(os.path.join(MODEL_DIR, "explainer.pkl"))
feat_names   = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

NUMERICAL   = ["age","trestbps","chol","thalach","oldpeak"]
CATEGORICAL = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
ALL_COLS    = NUMERICAL + CATEGORICAL

print("All model artifacts loaded successfully")

# ── Home route ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Predict route ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build a single-row DataFrame in the right column order
        patient = pd.DataFrame([{
            "age"     : float(data["age"]),
            "sex"     : int(data["sex"]),
            "cp"      : int(data["cp"]),
            "trestbps": float(data["trestbps"]),
            "chol"    : float(data["chol"]),
            "fbs"     : int(data["fbs"]),
            "restecg" : int(data["restecg"]),
            "thalach" : float(data["thalach"]),
            "exang"   : int(data["exang"]),
            "oldpeak" : float(data["oldpeak"]),
            "slope"   : int(data["slope"]),
            "ca"      : int(data["ca"]),
            "thal"    : int(data["thal"]),
        }])

        # Preprocess
        processed = preprocessor.transform(patient)

        # Predict
        prediction  = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1])

        # SHAP explanation
        shap_vals = explainer.shap_values(processed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]

        # Build top-6 explanation
        shap_dict = dict(zip(feat_names, shap_vals))
        top6 = sorted(shap_dict.items(),
                      key=lambda x: abs(x[1]), reverse=True)[:6]

        explanation = [
            {
                "feature"   : f,
                "shap_value": round(float(v), 4),
                "direction" : "risk" if v > 0 else "protective",
            }
            for f, v in top6
        ]

        # Risk band label
        if probability >= 0.75:
            risk_band = "High"
        elif probability >= 0.45:
            risk_band = "Moderate"
        else:
            risk_band = "Low"

        return jsonify({
            "prediction" : prediction,
            "probability": round(probability * 100, 1),
            "label"      : "Heart Disease Detected" if prediction == 1
                           else "No Heart Disease Detected",
            "risk_band"  : risk_band,
            "explanation": explanation,
            "status"     : "ok"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ── Health check (useful for Render deployment) ───────────
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "CardioSense v1"})
@app.route("/model-report")
def model_report():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score,
            confusion_matrix
        )
        import json

        X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
        X_test  = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

        with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
            best_name = f.read().strip()

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss", random_state=42),
        }

        report = []
        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)[:, 1]
            cm     = confusion_matrix(y_test, y_pred).tolist()
            report.append({
                "name"     : name,
                "accuracy" : round(float(accuracy_score(y_test, y_pred))  * 100, 1),
                "precision": round(float(precision_score(y_test, y_pred)) * 100, 1),
                "recall"   : round(float(recall_score(y_test, y_pred))    * 100, 1),
                "f1"       : round(float(f1_score(y_test, y_pred))        * 100, 1),
                "auc"      : round(float(roc_auc_score(y_test, y_prob)),  3),
                "cm"       : cm,
                "is_best"  : (name == best_name),
            })

        return jsonify({"status": "ok", "models": report, "best": best_name})

    except Exception as e:
        import traceback
        return jsonify({
            "status" : "error",
            "message": str(e),
            "detail" : traceback.format_exc()
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)