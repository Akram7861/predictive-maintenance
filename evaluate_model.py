from pathlib import Path
import json
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

test = pd.read_csv(ART / "test.csv")
target = "Engine Condition"

if target not in test.columns:
    raise ValueError(f"Target column missing: {target}")

X_test = test.drop(columns=[target])
y_test = test[target]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

pred = model.predict(X_test)

# Figure 1: Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Figure 1. Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrix.png", dpi=160)
plt.close()

# Figure 2: ROC curve for binary classification
if hasattr(model, "predict_proba") and len(model.classes_) == 2:
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Figure 2. Test Set ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG / "roc_curve.png", dpi=160)
    plt.close()

# Figure 3: Feature importance, when supported by the model
estimator = model
if hasattr(model, "named_steps"):
    estimator = model.named_steps.get("model", model)

if hasattr(estimator, "feature_importances_"):
    importance = pd.Series(
        estimator.feature_importances_,
        index=X_test.columns
    ).sort_values()

    ax = importance.plot(kind="barh", figsize=(8, 5))
    ax.set_title("Figure 3. Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance.png", dpi=160)
    plt.close()

metrics = {
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_precision_weighted": float(
        precision_score(y_test, pred, average="weighted", zero_division=0)
    ),
    "test_recall_weighted": float(
        recall_score(y_test, pred, average="weighted", zero_division=0)
    ),
    "test_f1_weighted": float(
        f1_score(y_test, pred, average="weighted", zero_division=0)
    )
}

if hasattr(model, "predict_proba") and len(model.classes_) == 2:
    metrics["test_roc_auc"] = float(
        roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    )

(ART / "evaluation_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

print("Evaluation completed successfully.")
print(json.dumps(metrics, indent=2))
print("Figures saved to:", FIG)
