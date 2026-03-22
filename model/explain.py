import numpy as np
import pandas as pd
import shap
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ── Load model + preprocessor ─────────────────────────────
model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

# ── Load raw test data (SHAP needs feature names) ─────────
X_test_raw = pd.read_csv(os.path.join(MODEL_DIR, "X_test_raw.csv"))
X_test     = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y_test     = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

print("Everything loaded successfully")
print("Test samples:", X_test_raw.shape[0])
# ── Reconstruct feature names after OHE ───────────────────
numerical_features   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg",
                        "exang", "slope", "ca", "thal"]

# Get the OHE-generated category names
ohe         = preprocessor.named_transformers_["cat"]["encoder"]
cat_names   = ohe.get_feature_names_out(categorical_features).tolist()
all_features = numerical_features + cat_names

print(f"\nTotal features after encoding: {len(all_features)}")
print("Feature names:")
for i, name in enumerate(all_features):
    print(f"  {i:>2}. {name}")
# ── Build SHAP explainer ──────────────────────────────────
# TreeExplainer works with Random Forest and XGBoost
# If model is Logistic Regression, use LinearExplainer instead
model_type = type(model).__name__
print(f"\nModel type: {model_type}")

if model_type == "LogisticRegression":
    explainer = shap.LinearExplainer(
        model, X_test,
        feature_perturbation="interventional"
    )
else:
    # Works for RandomForest and XGBoost
    explainer = shap.TreeExplainer(model)

# ── Compute SHAP values for entire test set ───────────────
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_test)

# For binary classifiers, shap_values may be a list [class0, class1]
# We want class 1 (disease present)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("SHAP values shape:", shap_values.shape)
print("Done computing SHAP values")

# ── Plot 1: Global feature importance ────────────────────
print("\nGenerating global importance plot...")

# Mean absolute SHAP value per feature
mean_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": mean_shap
}).sort_values("importance", ascending=True)

# Keep top 13 original-ish features for readability
importance_df = importance_df.tail(13)

plt.figure(figsize=(9, 6))
bars = plt.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="#4C9BE8",
    edgecolor="white",
    height=0.6
)
plt.xlabel("Mean |SHAP value|", fontsize=12)
plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "shap_global_importance.png"), dpi=150)
plt.close()
print("Saved: model/shap_global_importance.png")

# ── Function: explain a single prediction ─────────────────
def explain_prediction(patient_raw: pd.DataFrame):
    """
    Takes one raw patient row (as DataFrame),
    returns prediction + SHAP explanation as a dict.
    """
    # Preprocess
    patient_processed = preprocessor.transform(patient_raw)

    # Predict
    prediction  = int(model.predict(patient_processed)[0])
    probability = float(model.predict_proba(patient_processed)[0][1])

    # SHAP values for this patient
    patient_shap = explainer.shap_values(patient_processed)
    if isinstance(patient_shap, list):
        patient_shap = patient_shap[1]
    patient_shap = patient_shap[0]

    # Build explanation: top 6 features that influenced this prediction
    shap_dict = dict(zip(all_features, patient_shap))

    # Sort by absolute value — biggest impact first
    top_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:6]

    explanation = []
    for feat, shap_val in top_features:
        # Get the actual patient value for this feature
        orig_col = feat.split("_")[0]
        raw_val  = patient_raw[orig_col].values[0] \
                   if orig_col in patient_raw.columns else "N/A"

        explanation.append({
            "feature"   : feat,
            "shap_value": round(float(shap_val), 4),
            "direction" : "increases risk" if shap_val > 0 else "decreases risk",
            "raw_value" : str(raw_val)
        })

    return {
        "prediction" : prediction,
        "probability": round(probability, 4),
        "label"      : "Heart Disease Detected" if prediction == 1
                       else "No Heart Disease",
        "explanation": explanation
    }


# ── Test it on 3 real patients from test set ──────────────
print("\n" + "="*60)
print("TESTING EXPLAIN_PREDICTION ON 3 PATIENTS")
print("="*60)

for i in range(3):
    sample     = X_test_raw.iloc[[i]]
    result     = explain_prediction(sample)
    actual     = int(y_test[i])

    print(f"\nPatient {i+1}")
    print(f"  Actual    : {'Disease' if actual == 1 else 'No Disease'}")
    print(f"  Predicted : {result['label']}")
    print(f"  Confidence: {result['probability']*100:.1f}%")
    print(f"  Top reasons:")
    for exp in result["explanation"]:
        arrow = "▲" if exp["direction"] == "increases risk" else "▼"
        print(f"    {arrow} {exp['feature']:<20} "
              f"SHAP={exp['shap_value']:>7.4f}  "
              f"value={exp['raw_value']}")

# ── Save explainer and feature names ──────────────────────
joblib.dump(explainer,    os.path.join(MODEL_DIR, "explainer.pkl"))
joblib.dump(all_features, os.path.join(MODEL_DIR, "feature_names.pkl"))

# Also save explain_prediction as a standalone utility
# (we'll import it directly in app.py)
print("\nSaved:")
print("  model/explainer.pkl")
print("  model/feature_names.pkl")
print("  model/shap_global_importance.png")

print("\nDay 3 complete!")
print("Your model can now explain every single prediction it makes.")
