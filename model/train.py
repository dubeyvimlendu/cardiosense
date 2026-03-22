import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH        = os.path.join(BASE_DIR, "data", "heart.csv")
PREPROCESSOR_OUT = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
X_TRAIN_OUT      = os.path.join(BASE_DIR, "model", "X_train.npy")
X_TEST_OUT       = os.path.join(BASE_DIR, "model", "X_test.npy")
Y_TRAIN_OUT      = os.path.join(BASE_DIR, "model", "y_train.npy")
Y_TEST_OUT       = os.path.join(BASE_DIR, "model", "y_test.npy")
X_TRAIN_RAW_OUT  = os.path.join(BASE_DIR, "model", "X_train_raw.csv")
X_TEST_RAW_OUT   = os.path.join(BASE_DIR, "model", "X_test_raw.csv")

print("Project root:", BASE_DIR)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isnull().sum())
print("\nFirst row:\n", df.iloc[0])


numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
target = "target"

X = df.drop(columns=[target])
y = df[target]

print("\nFeature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nPreprocessor built successfully")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size :", X_test.shape[0])


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print("\nProcessed train shape:", X_train_processed.shape)
print("Processed test shape :", X_test_processed.shape)


os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

joblib.dump(preprocessor, PREPROCESSOR_OUT)
np.save(X_TRAIN_OUT, X_train_processed)
np.save(X_TEST_OUT,  X_test_processed)
np.save(Y_TRAIN_OUT, y_train.values)
np.save(Y_TEST_OUT,  y_test.values)
X_train.to_csv(X_TRAIN_RAW_OUT, index=False)
X_test.to_csv(X_TEST_RAW_OUT,   index=False)

print("\nAll files saved to model/ folder")


print("\n--- SANITY CHECK ---")
loaded_preprocessor = joblib.load(PREPROCESSOR_OUT)
sample = X_test.iloc[[0]]
processed_sample = loaded_preprocessor.transform(sample)
print("Raw sample:", sample.to_dict(orient="records")[0])
print("Processed shape:", processed_sample.shape)
