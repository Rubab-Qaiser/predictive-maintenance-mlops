import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import logging
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Fix 1: Proper logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retrain_model():
    """
    Automated retraining pipeline
    1. Generate new training data
    2. Train new model
    3. Evaluate against production
    4. Promote if better
    """
    logger.info('Starting automated retraining...')
    
    # Set MLflow
    try:
        mlflow.set_experiment('predictive-maintenance')
        client = MlflowClient()
        mlflow_available = True
    except Exception as e:
        logger.warning(f'MLflow not available: {e}')
        mlflow_available = False
    
    # Generate new training data (with drift)
    logger.info('Generating new training data...')
    n_samples = 10000
    np.random.seed(int(datetime.now().timestamp()))
    
    # Shifted distribution (simulating data drift)
    temperature = np.random.normal(85, 15, n_samples)  # Higher mean
    vibration = np.random.normal(0.6, 0.2, n_samples)  # Higher mean
    pressure = np.random.normal(110, 20, n_samples)
    rpm = np.random.normal(1500, 200, n_samples)
    age_days = np.random.randint(0, 365, n_samples)
    
    failure_score = (
        (temperature > 90) * 0.3 +
        (vibration > 0.8) * 0.3 +
        (pressure > 130) * 0.2 +
        (age_days > 300) * 0.2
    )
    failure_prob = failure_score + np.random.normal(0, 0.1, n_samples)
    failure = (failure_prob > 0.5).astype(int)
    
    data = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'rpm': rpm,
        'age_days': age_days,
        'failure': failure
    })
    
    logger.info(f'New data: {len(data)} samples, {data.failure.mean():.2%} failure rate')
    
    # Prepare data
    X = data.drop('failure', axis=1)
    y = data['failure']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train new model
    logger.info('Training new model...')
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Fix 2: Convert to Python float explicitly
    new_accuracy = float(accuracy_score(y_test, y_pred))
    new_roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    
    logger.info(f'New model - Accuracy: {new_accuracy:.4f}, ROC AUC: {new_roc_auc:.4f}')
    
    # Try MLflow if available
    if mlflow_available:
        try:
            with mlflow.start_run(run_name='auto_retrain') as run:
                mlflow.log_param('retrain_date', datetime.now().isoformat())
                mlflow.log_param('n_samples', n_samples)
                mlflow.log_metric('accuracy', new_accuracy)
                mlflow.log_metric('roc_auc', new_roc_auc)
                mlflow.sklearn.log_model(model, 'model')
    
    # Use the run object directly (not active_run())
                model_uri = f'runs:/{run.info.run_id}/model'
                mlflow.register_model(model_uri=model_uri, name='PredictiveMaintenance')
                logger.info('Model registered to MLflow')
        except Exception as e:
            logger.warning(f'MLflow logging failed: {e}')
            mlflow_available = False
    
    # Manual save as fallback
    if not mlflow_available:
        logger.info('Saving model manually...')
        os.makedirs('model_registry', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'model_registry/PredictiveMaintenance_retrained_{timestamp}.pkl'
        joblib.dump(model, model_path)
        logger.info(f'Model saved manually to: {model_path}')
    
    # Compare with existing production model
    prod_model_path = 'model_registry/PredictiveMaintenance_v1.pkl'
    if os.path.exists(prod_model_path):
        try:
            prod_model = joblib.load(prod_model_path)
            prod_pred = prod_model.predict(X_test_scaled)
            prod_accuracy = float(accuracy_score(y_test, prod_pred))
            logger.info(f'Production model - Accuracy: {prod_accuracy:.4f}')
            
            if new_accuracy > prod_accuracy:
                logger.info(f'✅ New model is BETTER! Promoting to production...')
                # Save as new version
                new_version = 2
                new_model_path = f'model_registry/PredictiveMaintenance_v{new_version}.pkl'
                joblib.dump(model, new_model_path)
                logger.info(f'New model saved as version {new_version}')
            else:
                logger.info(f'New model not better than production. No promotion.')
        except Exception as e:
            logger.warning(f'Error comparing with production model: {e}')
    else:
        logger.info('No production model found. Saving as production candidate.')
        joblib.dump(model, prod_model_path)
    
    return new_roc_auc

if __name__ == '__main__':
    retrain_model()