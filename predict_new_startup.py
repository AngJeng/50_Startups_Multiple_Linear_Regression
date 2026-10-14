"""Load the trained model and predict profit for a new startup."""

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "startup_profit_model.joblib"
BASELINE_PATH = ROOT / "models" / "baseline_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run `python main.py` first."
    )

model = joblib.load(MODEL_PATH)

new_startup = pd.DataFrame(
    {
        "R&D Spend": [100000],
        "Administration": [120000],
        "Marketing Spend": [250000],
        "State": ["New York"],
    }
)

prediction = model.predict(new_startup)[0]
print("New Startup Profile:")
print(new_startup.to_string(index=False))
print(f"\nPredicted Profit (Best Feature-Selected Model): ${prediction:,.2f}")

if BASELINE_PATH.exists():
    baseline_model = joblib.load(BASELINE_PATH)
    base_pred = baseline_model.predict(new_startup)[0]
    print(f"Predicted Profit (Baseline All-Features Model):  ${base_pred:,.2f}")
