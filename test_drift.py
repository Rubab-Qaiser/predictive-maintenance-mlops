import numpy as np
import pandas as pd
from drift_detector import DriftDetector

# Generate reference data (training distribution)
np.random.seed(42)
n_samples = 1000

reference_data = pd.DataFrame({
    'temperature': np.random.normal(75, 15, n_samples),
    'vibration': np.random.normal(0.5, 0.2, n_samples),
    'pressure': np.random.normal(100, 20, n_samples)
})

# Initialize detector
detector = DriftDetector(reference_data)

# Test 1: No drift (same distribution)
print('Test 1: No Drift Expected')
current_data_no_drift = pd.DataFrame({
    'temperature': np.random.normal(75, 15, 500),
    'vibration': np.random.normal(0.5, 0.2, 500),
    'pressure': np.random.normal(100, 20, 500)
})

drift, results = detector.check_all_features(current_data_no_drift)
print(f'Drift Detected: {drift}\n')

# Test 2: With drift (shifted distribution)
print('Test 2: Drift Expected')
current_data_drift = pd.DataFrame({
    'temperature': np.random.normal(95, 15, 500),  # Mean shifted!
    'vibration': np.random.normal(0.8, 0.2, 500),  # Mean shifted!
    'pressure': np.random.normal(100, 20, 500)
})

drift, results = detector.check_all_features(current_data_drift)
print(f'Drift Detected: {drift}')

print('\nDrift Details:')
for feature, result in results.items():
    if result['drift_detected']:
        print(f"  {feature}: DRIFT (p={result['p_value']:.4f})")

# Save report
detector.save_report()
print('\nDrift report saved to drift_report.json')