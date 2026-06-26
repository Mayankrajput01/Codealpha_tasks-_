# ================================================================
#  Heart Disease Prediction — train_model.py
#  Model: Gradient Boosting — well tuned
# ================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "meta.pkl")

print("=" * 60)
print("   Heart Disease Prediction — Training Started")
print("=" * 60)

# ── Step 1: Generate realistic dataset ───────────────────────
# Based on UCI Cleveland Heart Disease dataset patterns
np.random.seed(42)
N = 2000

# Healthy people (no disease)
n_healthy = N // 2
healthy = {
    "age"     : np.random.randint(30, 65, n_healthy),
    "sex"     : np.random.choice([0, 1], n_healthy, p=[0.55, 0.45]),
    "cp"      : np.random.choice([1, 2, 3], n_healthy, p=[0.5, 0.3, 0.2]),
    "trestbps": np.random.randint(100, 140, n_healthy),
    "chol"    : np.random.randint(150, 240, n_healthy),
    "fbs"     : np.random.choice([0, 1], n_healthy, p=[0.85, 0.15]),
    "restecg" : np.random.choice([0, 1, 2], n_healthy, p=[0.5, 0.4, 0.1]),
    "thalach" : np.random.randint(140, 190, n_healthy),
    "exang"   : np.random.choice([0, 1], n_healthy, p=[0.8, 0.2]),
    "oldpeak" : np.random.uniform(0.0, 1.5, n_healthy).round(1),
    "slope"   : np.random.choice([1, 2, 3], n_healthy, p=[0.5, 0.4, 0.1]),
    "ca"      : np.random.choice([0, 1, 2, 3], n_healthy, p=[0.7, 0.2, 0.07, 0.03]),
    "thal"    : np.random.choice([3, 6, 7], n_healthy, p=[0.7, 0.1, 0.2]),
    "target"  : np.zeros(n_healthy, dtype=int),
}

# Diseased people (heart disease)
n_disease = N - n_healthy
disease = {
    "age"     : np.random.randint(45, 75, n_disease),
    "sex"     : np.random.choice([0, 1], n_disease, p=[0.3, 0.7]),
    "cp"      : np.random.choice([1, 2, 3, 4], n_disease, p=[0.1, 0.15, 0.2, 0.55]),
    "trestbps": np.random.randint(130, 180, n_disease),
    "chol"    : np.random.randint(220, 380, n_disease),
    "fbs"     : np.random.choice([0, 1], n_disease, p=[0.65, 0.35]),
    "restecg" : np.random.choice([0, 1, 2], n_disease, p=[0.2, 0.4, 0.4]),
    "thalach" : np.random.randint(90, 155, n_disease),
    "exang"   : np.random.choice([0, 1], n_disease, p=[0.35, 0.65]),
    "oldpeak" : np.random.uniform(1.0, 5.0, n_disease).round(1),
    "slope"   : np.random.choice([1, 2, 3], n_disease, p=[0.1, 0.35, 0.55]),
    "ca"      : np.random.choice([0, 1, 2, 3], n_disease, p=[0.2, 0.3, 0.3, 0.2]),
    "thal"    : np.random.choice([3, 6, 7], n_disease, p=[0.15, 0.2, 0.65]),
    "target"  : np.ones(n_disease, dtype=int),
}

df_h = pd.DataFrame(healthy)
df_d = pd.DataFrame(disease)
df   = pd.concat([df_h, df_d], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset: {len(df)} samples")
print(f"  No Disease : {(df['target']==0).sum()}")
print(f"  Disease    : {(df['target']==1).sum()}")

# ── Step 2: Features & target ────────────────────────────────
FEATURES = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]

X = df[FEATURES].values
y = df["target"].values

# ── Step 3: Split & scale ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Step 4: Train (well-tuned Gradient Boosting) ─────────────
print("\nTraining model...")
model = GradientBoostingClassifier(
    n_estimators     = 300,
    learning_rate    = 0.07,
    max_depth        = 4,
    min_samples_leaf = 10,
    subsample        = 0.85,
    random_state     = 42
)
model.fit(X_train, y_train)

# ── Step 5: Evaluate ─────────────────────────────────────────
y_pred    = model.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"CV  Accuracy  : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease","Disease"]))

# ── Step 6: Save ─────────────────────────────────────────────
meta = {
    "features" : FEATURES,
    "label_map": {0: "No Disease", 1: "Disease"},
}

with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
with open(META_PATH,   "wb") as f: pickle.dump(meta,   f)

print(f"\nModel  -> {MODEL_PATH}")
print(f"Scaler -> {SCALER_PATH}")
print(f"Meta   -> {META_PATH}")
print("\n✅ Training complete! Run: streamlit run app.py")
