import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"


def load_raw_data(filename: str = "Invistico_Airline.csv") -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / filename)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / name)
