from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "engine_data.csv"

EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

if not DATA_PATH.exists():
    print(f"ERROR: Dataset not found: {DATA_PATH}", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
if missing_columns:
    print(f"ERROR: Missing expected columns: {missing_columns}", file=sys.stderr)
    print(f"Actual columns: {df.columns.tolist()}", file=sys.stderr)
    sys.exit(1)

print("DATA VALIDATION PASSED")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes.to_string())
print("\nMissing values:")
print(df.isnull().sum().to_string())
print(f"\nDuplicate rows: {df.duplicated().sum():,}")
print("\nNumeric summary:")
print(df.describe().round(2).to_string())

print("\nTarget distribution:")
print(df["Engine Condition"].value_counts(dropna=False).to_string())
