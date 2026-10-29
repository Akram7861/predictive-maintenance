from pathlib import Path
import json
import pickle
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODELS = ROOT / "models"
ART.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(ART / "train.csv")
test = pd.read_csv(ART / "test.csv")
target = "Engine Condition"

if target not in train.columns or target not in test.columns:
    raise ValueError(f"Target column '{target}' missing from train/test data")

X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.drop(columns=[target])
y_test = test[target]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test feature columns do not match")

# Baseline comparison
candidates = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=42,
        class_weight="balanced", n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

rows = []
for name, estimator in candidates.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, pred, average="weighted", zero_division=0
        )
    })

comparison = pd.DataFrame(rows).sort_values(
    "f1_weighted", ascending=False
)
comparison.to_csv(ART / "model_comparison.csv", index=False)
print("Baseline model comparison:")
print(comparison.round(4).to_string(index=False))

# Tune Random Forest with a modest grid
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 3]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_weighted",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Final test-set evaluation
pred = best_model.predict(X_test)
metrics = {
    "selected_model": "Random Forest",
    "best_parameters": search.best_params_,
    "best_cv_f1_weighted": float(search.best_score_),
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_precision_weighted": float(
        precision_score(y_test, pred, average="weighted", zero_division=0)
    ),
    "test_recall_weighted": float(
        recall_score(y_test, pred, average="weighted", zero_division=0)
    ),
    "test_f1_weighted": float(
        f1_score(y_test, pred, average="weighted", zero_division=0)
    ),
    "classification_report": classification_report(
        y_test, pred, output_dict=True, zero_division=0
    ),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

if hasattr(best_model, "predict_proba") and len(best_model.classes_) == 2:
    metrics["test_roc_auc"] = float(
        roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    )

with open(MODELS / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

(ART / "best_model_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)
(ART / "best_model_params.json").write_text(
    json.dumps(search.best_params_, indent=2)
)
pd.DataFrame(search.cv_results_).to_csv(
    ART / "hyperparameter_tuning_results.csv", index=False
)

print("\nBest parameters:", search.best_params_)
print("Best cross-validation weighted F1:", round(search.best_score_, 4))
print("\nFinal test metrics:")
for key in [
    "test_accuracy", "test_precision_weighted",
    "test_recall_weighted", "test_f1_weighted",
    "test_roc_auc"
]:
    if key in metrics:
        print(f"{key}: {metrics[key]:.4f}")

print("\nSaved model:", MODELS / "best_model.pkl")
print("Saved metrics:", ART / "best_model_metrics.json")
