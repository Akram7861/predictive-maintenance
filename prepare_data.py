from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "engine_data.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
target = "Engine Condition"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Columns: {df.columns.tolist()}"
    )

if df[target].isna().any():
    raise ValueError("Target column contains missing values.")

df = df.drop_duplicates().copy()

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train = X_train.copy()
train[target] = y_train

test = X_test.copy()
test[target] = y_test

train.to_csv(ARTIFACTS / "train.csv", index=False)
test.to_csv(ARTIFACTS / "test.csv", index=False)

print("Data preparation successful.")
print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")
print(f"Saved: {ARTIFACTS / 'train.csv'}")
print(f"Saved: {ARTIFACTS / 'test.csv'}")
