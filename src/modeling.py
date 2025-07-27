import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from typing import Tuple, Optional

def load_data(filepath: str = "data/processed/final_model_input.csv", target_col: str = "class") -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

def train_and_select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = "weighted_recall",
    force_lightgbm: bool = False
) -> Tuple[str, object, float, pd.Series]:

    models = get_models()
    best_model = None
    best_score = 0
    best_name = None

    scoring_func = {
        "weighted_recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    }

    if metric not in scoring_func:
        raise ValueError(f"Unsupported metric {metric}. Use one of {list(scoring_func.keys())}")

    scorer = scoring_func[metric]

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = scorer(y_test, preds)
        print(f"{name} {metric}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    if force_lightgbm:
        print("\n⚠️  Forcing LightGBM as the selected model.")
        best_model = LGBMClassifier()
        best_model.fit(X_train, y_train)
        best_name = "LightGBM (Forced)"
        preds = best_model.predict(X_test)
        best_score = scorer(y_test, preds)

    else:
        preds = best_model.predict(X_test)

    return best_name, best_model, best_score, preds

def save_model(model, model_path="outputs/models/best_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def save_predictions(predictions, pred_path="outputs/predictions/test_predictions.csv"):
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(pred_path, index=False)
    print(f"Predictions saved at {pred_path}")

def save_classification_report(y_true, y_pred, report_path="outputs/metrics/classification_report.json"):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_json(report_path, indent=4)
    print(f"Classification report saved at {report_path}")

def main(force_lightgbm: Optional[bool] = False):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_name, best_model, best_score, preds = train_and_select_best_model(
        X_train, y_train, X_test, y_test, metric="weighted_recall", force_lightgbm=force_lightgbm
    )

    print(f"\n✅ Final model: {best_name} (Score: {best_score:.4f})")

    save_model(best_model)
    save_predictions(preds)
    save_classification_report(y_test, preds)

    print("📁 Artifacts saved under /outputs/")

if __name__ == "__main__":
    main()
