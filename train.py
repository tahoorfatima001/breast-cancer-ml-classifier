"""
Breast Cancer Diagnosis Classifier - training & evaluation script.

Workflow: load data -> explore -> split -> compare models with cross-validation
-> evaluate the best model on unseen data -> save model, metrics and charts.

Run:  python train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, classification_report)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------- 1. Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")          # 0 = malignant, 1 = benign
class_names = list(data.target_names)               # ['malignant', 'benign']
print("Dataset shape:", X.shape)
print("Missing values:", int(X.isnull().sum().sum()))
print("Class counts:", dict(zip(class_names, np.bincount(y))))

# ---------------------------------------------------------------- 2. Explore
plt.figure(figsize=(5, 4))
counts = y.map(dict(enumerate(class_names))).value_counts()
plt.bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
plt.title("Class Distribution")
plt.ylabel("Number of samples")
plt.tight_layout(); plt.savefig("images/class_distribution.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ---------------------------------------------------------------- 4. Compare models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=SEED)),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, pipe in models.items():
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV Accuracy": cv_acc,
        "Test Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, proba),
    })
results = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=False).reset_index(drop=True)
print("\n", results.round(4).to_string(index=False))
results.round(4).to_csv("results.csv", index=False)

plt.figure(figsize=(8, 4.5))
plt.bar(results["Model"], results["Test Accuracy"], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.ylim(0.9, 1.0); plt.ylabel("Test accuracy"); plt.title("Model Comparison")
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. Best model
best_name = results.loc[0, "Model"]            # chosen using cross-validation only
best = models[best_name]
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
print(f"\nBest model (by CV accuracy): {best_name}")
print(classification_report(y_test, pred, target_names=class_names))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=class_names).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}"); plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, lw=2, label=f"{best_name} (AUC = {roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC Curve"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. Feature importance
rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top = imp.head(10)
plt.figure(figsize=(8, 5))
top[::-1].plot(kind="barh", color="#4C72B0")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7. Save
joblib.dump(best, "model/best_model.joblib")
meta = {
    "best_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "top_features": list(top.index),
    "feature_names": list(X.columns),
    "feature_means": X.mean().round(4).to_dict(),
    "feature_min": X.min().round(4).to_dict(),
    "feature_max": X.max().round(4).to_dict(),
    "class_names": class_names,
}
json.dump(meta, open("model/metadata.json", "w"), indent=2)
print("\nSaved: model/best_model.joblib, model/metadata.json, results.csv, images/*.png")
