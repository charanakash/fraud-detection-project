import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import yaml
from pathlib import Path

# Import our custom modules
from fraud_detection.data_processing import load_data, create_pipeline, split_data
from fraud_detection.evaluate import evaluate_model

def train():
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
    
    # Load Configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Set up the MLflow Experiment
    mlflow.set_experiment("Fraud Detection")

    # Start a new MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # 1. Load and Split Data
        raw_data_path = PROJECT_ROOT / config['data']['raw_path']
        df = load_data(raw_data_path)
        X_train, X_test, y_train, y_test = split_data(
            df,
            config['model']['target_column'],
            config['model']['test_size'],
            config['model']['random_state']
        )

        # Log general parameters
        mlflow.log_param("test_size", config['model']['test_size'])
        mlflow.log_param("random_state", config['model']['random_state'])
        
        # 2. Create Preprocessing Pipeline
        preprocessor = create_pipeline()
        cols_to_scale = ['Time', 'Amount']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[cols_to_scale] = preprocessor.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = preprocessor.transform(X_test[cols_to_scale])

        # 3. Define Models and log their specific parameters
        models = {
            "RandomForest": RandomForestClassifier(**config['model']['random_forest']),
            "XGBoost": XGBClassifier(**config['model']['xgboost'], use_label_encoder=False, eval_metric='logloss')
        }

        # Log parameters with a model-specific prefix to avoid key collisions
        print("Logging model parameters to MLflow...")
        for model_name, params in config['model'].items():
            if isinstance(params, dict): # Ensures we only log dicts like 'xgboost', not 'save_path'
                prefixed_params = {f"{model_name}_{key}": value for key, value in params.items()}
                mlflow.log_params(prefixed_params)

        best_model = None
        best_score = 0
        best_model_name = ""

        # 4. Train and Evaluate Models
        for name, model in models.items():
            print(f"--- Training {name} ---")
            model.fit(X_train_scaled, y_train)
            
            print(f"--- Evaluating {name} ---")
            # NOTE: Assumes evaluate_model now returns a dictionary of metrics 
            # e.g., {'auc_pr': 0.83, 'f1_score': 0.80}
            metrics = evaluate_model(model, X_test_scaled, y_test)
            
            # Log metrics with a prefix for clarity in the UI
            prefixed_metrics = {f"{name}_{key}": value for key, value in metrics.items()}
            mlflow.log_metrics(prefixed_metrics)
            
            if metrics.get('auc_pr', 0) > best_score:
                best_score = metrics['auc_pr']
                best_model = model
                best_model_name = name
                print(f"New best model: {name} with AUC-PR: {best_score:.4f}")   
        
        print(f"\nSelected best model: {best_model_name}")

        # 5. Log the best model pipeline to MLflow Model Registry
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])

        # Log the full pipeline, which includes preprocessing
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model_pipeline", # This is the folder name within the run's artifacts
            registered_model_name="fraud-detection-model" # This is the name in the Model Registry
        )

        # 6. Log other artifacts, like the evaluation plot
        # Assumes evaluate_model creates this file
        mlflow.log_artifact("precision_recall_curve.png")
        
        print(f"\n✅ Best model pipeline logged and registered to MLflow.")

if __name__ == "__main__":
    train()