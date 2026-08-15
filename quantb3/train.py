"""CLI reprodutível para criar dataset, ajustar modelos e salvar o pipeline principal.

Uso: python -m quantb3.train
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantb3.config import ARTIFACT_DIR, END_DATE, MODEL_DIR, START_DATE, TICKERS
from quantb3.data import build_dataset
from quantb3.modeling import (classification_metrics, feature_columns, optimize_threshold,
                               performance_metrics, strategy_backtest, temporal_splits,
                               walk_forward_auc)


def main() -> None:
    """Executa treino, tuning de 50 trials e persiste resultados reprodutíveis."""
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    dataset, _, benchmark = build_dataset(TICKERS, START_DATE, END_DATE)
    columns = feature_columns(dataset)
    train, validation, test = temporal_splits(dataset)
    train = train.dropna(subset=columns)
    validation = validation.dropna(subset=columns)
    test = test.dropna(subset=columns)
    imbalance = train["target"].value_counts(normalize=True).to_dict()
    print(f"Proporção de classes no treino: {imbalance}")
    # class_weight é temporalmente seguro; SMOTE pode sintetizar pontos usando padrões futuros.
    baseline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000))])
    forest = Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1))])
    for name, model in {"logistic": baseline, "random_forest": forest}.items():
        model.fit(train[columns], train["target"])
        p = model.predict_proba(validation[columns])[:, 1]
        print(name, classification_metrics(validation["target"], p, 0.5))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbosity": -1,
        }
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", lgb.LGBMClassifier(**params))])
        return float(np.mean(walk_forward_auc(pipeline, train, columns)))

    study = optuna.create_study(direction="maximize", study_name="quantb3_lgbm")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", lgb.LGBMClassifier(**study.best_params, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1))])
    pipeline.fit(train[columns], train["target"])
    validation_probability = pipeline.predict_proba(validation[columns])[:, 1]
    threshold = optimize_threshold(validation["target"], validation_probability)
    test_probability = pipeline.predict_proba(test[columns])[:, 1]
    metrics = classification_metrics(test["target"], test_probability, threshold)
    backtest = strategy_backtest(test, test_probability, threshold, benchmark)
    metrics.update(performance_metrics(backtest))
    metrics["threshold"] = threshold
    joblib.dump(pipeline, MODEL_DIR / "pipeline_lgbm.pkl")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(columns, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    backtest.to_csv(ARTIFACT_DIR / "backtest.csv")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
