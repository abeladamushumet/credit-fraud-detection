import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb  

# === CONFIG ===
model_path = "outputs/models/best_model.pkl"  
data_path = "data/processed/final_model_input.csv"
target_col = "class"
output_dir = "figures"

os.makedirs(output_dir, exist_ok=True)

def load_model_and_data(model_path, data_path):
    print("🔄 Loading model and data...")
    # Load LightGBM model saved by joblib (LGBMClassifier)
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    print(f"✅ Model and data loaded. Data shape: {X.shape}")
    return model, X

def plot_feature_importance(model, X):
    # LightGBM classifier has feature_importances
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align="center")
    plt.yticks(range(len(indices)), X.columns[indices][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    print(f"✅ Feature importance saved to: {output_dir}/feature_importance.png")

def plot_shap_summary(model, X):
    print("🔄 Starting SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[:50])
    print("✅ SHAP values computed.")
    plt.figure()
    shap.summary_plot(shap_values, X.iloc[:50], max_display=20, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    print(f"✅ SHAP summary plot saved to: {output_dir}/shap_summary.png")

def plot_waterfall(model, X):
    print("🔄 Starting local waterfall explanation...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[0]])
    expected_value = explainer.expected_value

    # Handle binary classification shap_values output format
    if isinstance(shap_values, list):
        shap_val_for_plot = shap_values[1][0]
        expected_value = explainer.expected_value[1]
    else:
        shap_val_for_plot = shap_values[0]
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]

    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value=expected_value,
        shap_values=shap_val_for_plot,
        features=X.iloc[0],
        max_display=20,
        show=False
    )
    plt.savefig(os.path.join(output_dir, "local_waterfall.png"))
    print(f"✅ Local waterfall plot saved to: {output_dir}/local_waterfall.png")

def plot_force_plot(model, X):
    print("🔄 Starting SHAP force plot...")
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X.iloc[[0]])

        force_plot = shap.plots.force(
            shap_values.base_values[0],
            shap_values.values[0],
            X.iloc[0],
            matplotlib=False,
            show=False
        )

        shap.save_html(os.path.join(output_dir, "force_plot.html"), force_plot)
        print(f"✅ SHAP force plot saved to: {output_dir}/force_plot.html")
    except Exception as e:
        print(f"❌ Error creating force plot: {e}")

if __name__ == "__main__":
    model, X = load_model_and_data(model_path, data_path)
    plot_feature_importance(model, X)
    plot_shap_summary(model, X)
    plot_waterfall(model, X)
    plot_force_plot(model, X)
