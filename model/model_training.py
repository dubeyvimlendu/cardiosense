import numpy as np
import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_OUT    = os.path.join(MODEL_DIR, "best_model.pkl")

# ── Load preprocessed data ────────────────────────────────
X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

print("Data loaded successfully")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# ── Define models ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# ── Cross-validation setup ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*60)

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="roc_auc"
    )
    cv_results[name] = scores.mean()
    print(f"\n{name}")
    print(f"  AUC per fold : {scores.round(3)}")
    print(f"  Mean AUC     : {scores.mean():.4f}")
    print(f"  Std          : {scores.std():.4f}")
# ── Train each model on full training data ────────────────
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

results = {}

for name, model in models.items():
    # Train on full training set
    model.fit(X_train, y_train)

    # Predict
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    results[name] = {
        "model"    : model,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
        "auc"      : roc_auc_score(y_test, y_pred_prob),
    }

    r = results[name]
    print(f"\n{name}")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1 Score  : {r['f1']:.4f}")
    print(f"  ROC-AUC   : {r['auc']:.4f}")

# ── Pick best model by AUC ────────────────────────────────
best_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_name]["model"]

print("\n" + "="*60)
print(f"WINNER: {best_name}")
print(f"AUC Score: {results[best_name]['auc']:.4f}")
print("="*60)

# ── Detailed report for best model ───────────────────────
print(f"\nDetailed report — {best_name}")
y_pred_best = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best,
      target_names=["No Disease", "Disease"]))

# ── Comparison table ──────────────────────────────────────
print("\nMODEL COMPARISON SUMMARY")
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print("-" * 58)
for name, r in results.items():
    marker = " <-- BEST" if name == best_name else ""
    print(f"{name:<25} {r['accuracy']:>10.4f} "
          f"{r['f1']:>10.4f} {r['auc']:>10.4f}{marker}")
# ── Save best model ───────────────────────────────────────
joblib.dump(best_model, MODEL_OUT)

# Save best model name as text for reference
with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_name)

print(f"\nBest model saved → model/best_model.pkl")
print(f"Model name saved → model/best_model_name.txt")
print("\nDay 2 complete!")
