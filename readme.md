<h1 align="center">CardioSense — Heart Disease Risk Predictor</h1>

<p align="center">
  <i>ML-powered heart disease risk assessment with per-patient 
  SHAP explainability — built end-to-end and deployed.</i>
</p>

<p align="center">
  <b>Live Demo:</b> https://web-production-ee5df.up.railway.app/ &nbsp;|&nbsp;
  <b>Dataset:</b> UCI Heart Disease (303 patients)
</p>

---

### Overview

CardioSense predicts heart disease risk from 13 patient vitals and explains
every prediction using SHAP values — showing exactly which factors 
increased or decreased the risk. Not just a binary yes/no.

---

### Key Features

- **Risk prediction** with confidence percentage and High/Moderate/Low band
- **SHAP explainability** — per-patient feature contribution breakdown
- **3 models compared** — Logistic Regression, Random Forest, XGBoost
- **REST API** — Flask backend serving JSON predictions
- **Deployed live** on Railway

---

### Model Performance

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 85.2% | 0.863 | 0.920 |
| Random Forest | 86.9% | 0.875 | 0.931 |
| **XGBoost** | **88.5%** | **0.897** | **0.944** |

XGBoost selected for highest ROC-AUC and Recall — the metrics that 
matter most for medical binary classification.

---

### Tech Stack

| Component | Technology |
|---|---|
| ML model | XGBoost |
| Explainability | SHAP TreeExplainer |
| Preprocessing | Sklearn Pipeline |
| API | Flask |
| Deployment | Railway |

---

### Run Locally
```bash
git clone https://github.com/dubeyvimlendu/cardiosense
cd cardiosense
pip install -r requirements.txt
python model/train.py
python model/model_training.py
python model/explain.py
python app.py
```

---

### API
```
POST /predict      → prediction + SHAP explanation
GET  /model-report → all 3 model metrics as JSON
GET  /health       → health check
```

---

### Author

Built by **Vimlendu Dubey**  
📧 dubeyvimlendu@gmail.com  
💼 github.com/dubeyvimlendu
