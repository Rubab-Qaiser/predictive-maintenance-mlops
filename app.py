from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title='Predictive Maintenance API',
    description='ML API for equipment failure prediction',
    version='1.0.0'
)

# Load production model from MANUAL registry
MODEL_NAME = 'PredictiveMaintenance'
MODEL_VERSION = 1
MODEL_PATH = f'model_registry/{MODEL_NAME}_v{MODEL_VERSION}.pkl'

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f'Loaded model from: {MODEL_PATH}')
    else:
        logger.error(f'Model file not found: {MODEL_PATH}')
        model = None
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    model = None

# Metrics storage
metrics = {
    'total_requests': 0,
    'predictions': [],
    'latencies': [],
    'failures_predicted': 0,
    'errors': 0
}

# Request model
class PredictionRequest(BaseModel):
    temperature: float = Field(..., description='Temperature in °F')
    vibration: float = Field(..., description='Vibration in mm/s')
    pressure: float = Field(..., description='Pressure in PSI')
    rpm: float = Field(..., description='Rotations per minute')
    age_days: int = Field(..., description='Days since last maintenance')
    
    class Config:
        json_schema_extra = {
            'example': {
                'temperature': 85.0,
                'vibration': 0.7,
                'pressure': 110.0,
                'rpm': 1500.0,
                'age_days': 200
            }
        }

# Response model
class PredictionResponse(BaseModel):
    will_fail: bool
    probability: float
    recommendation: str
    latency_ms: float
    timestamp: str

# Root endpoint
@app.get('/')
def root():
    return {
        'message': 'Predictive Maintenance API',
        'version': '1.0.0',
        'docs': '/docs',
        'model_loaded': model is not None
    }

# Health check endpoint
@app.get('/health')
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {'status': 'healthy', 'model_loaded': True}

# Prediction endpoint (ONLY ONE - KEEP THIS VERSION)
@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'temperature': request.temperature,
            'vibration': request.vibration,
            'pressure': request.pressure,
            'rpm': request.rpm,
            'age_days': request.age_days
        }])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Get probability
        try:
            proba = model.predict_proba(input_data)[0]
            probability = float(proba[1])
        except:
            probability = 1.0 if prediction == 1 else 0.0
        
        will_fail = bool(prediction == 1)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        metrics['total_requests'] += 1
        metrics['latencies'].append(latency_ms)
        metrics['predictions'].append(probability)
        if will_fail:
            metrics['failures_predicted'] += 1
        
        # Log prediction
        logger.info(f'Prediction: {will_fail}, Prob: {probability:.3f}, Latency: {latency_ms:.2f}ms')
        
        return PredictionResponse(
            will_fail=will_fail,
            probability=round(probability, 3),
            recommendation='Schedule maintenance' if will_fail else 'Continue operation',
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        metrics['errors'] += 1
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get('/metrics')
def get_metrics():
    avg_latency = sum(metrics['latencies']) / len(metrics['latencies']) if metrics['latencies'] else 0
    failure_rate = metrics['failures_predicted'] / metrics['total_requests'] if metrics['total_requests'] > 0 else 0
    
    return {
        'total_requests': metrics['total_requests'],
        'failures_predicted': metrics['failures_predicted'],
        'failure_rate': round(failure_rate, 3),
        'avg_latency_ms': round(avg_latency, 2),
        'errors': metrics['errors']
    }