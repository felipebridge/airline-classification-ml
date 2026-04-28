import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from utils import load_raw_data, save_model
from data_preprocessing import prepare, build_preprocessor
from evaluate_model import score, print_metrics


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "random_forest":       RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "gradient_boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
}


def train(df):
    X_train, X_test, y_train, y_test, le = prepare(df)
    target_names = list(le.classes_)

    results   = {}
    pipelines = {}

    for name, clf in MODELS.items():
        print(f"\n--- {name} ---")
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        metrics = score(pipe, X_test, y_test, target_names)
        print_metrics(name, metrics)
        results[name]   = metrics
        pipelines[name] = pipe

    best = max(results, key=lambda n: results[n]["roc_auc"])
    save_model(pipelines[best], f"best_model_{best}.joblib")
    print(f"\nBest model: {best} (ROC-AUC: {results[best]['roc_auc']:.4f})")

    return pipelines, results, best, X_test, y_test, target_names


if __name__ == "__main__":
    df = load_raw_data()
    train(df)
