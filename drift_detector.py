import numpy as np
import pandas as pd
from scipy import stats
import json
from datetime import datetime


class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        """
        reference_data : Training/reference dataset
        threshold      : p-value threshold
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history = []

    def detect_drift_ks(self, current_data, feature):
        """
        Kolmogorov-Smirnov drift detection

        Returns:
            drift_detected (bool)
            p_value (float)
            statistic (float)
        """

        ref_values = self.reference_data[feature].dropna().values
        curr_values = current_data[feature].dropna().values

        if len(ref_values) < 2 or len(curr_values) < 2:
            return False, 1.0, 0.0

        try:
            result = stats.ks_2samp(ref_values, curr_values)

            
            statistic = float(result[0])
            p_value = float(result[1])

            drift_detected = p_value < self.threshold

            return drift_detected, p_value, statistic

        except Exception as e:
            print(f"Error in KS test for feature '{feature}': {e}")
            return False, 1.0, 0.0

    def check_all_features(self, current_data):
        """
        Check drift across all features
        """

        results = {}
        drift_detected_any = False

        for feature in self.reference_data.columns:

            if feature not in current_data.columns:
                print(f"Warning: Feature '{feature}' not found")
                continue

            drift, p_val, stat = self.detect_drift_ks(
                current_data,
                feature
            )

            results[feature] = {
                "drift_detected": bool(drift),
                "p_value": float(p_val),
                "statistic": float(stat)
            }

            if drift:
                drift_detected_any = True

        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "drift_detected": bool(drift_detected_any),
            "features": results
        })

        return drift_detected_any, results

    def save_report(self, filename="drift_report.json"):

        with open(filename, "w") as f:
            json.dump(self.drift_history, f, indent=2)

        print(f" Drift report saved to {filename}")


if __name__ == "__main__":

    print("=" * 60)
    print("DRIFT DETECTOR TEST")
    print("=" * 60)

    np.random.seed(42)

    n_samples = 1000

    reference_data = pd.DataFrame({
        "temperature": np.random.normal(75, 15, n_samples),
        "vibration": np.random.normal(0.5, 0.2, n_samples),
        "pressure": np.random.normal(100, 20, n_samples)
    })

    detector = DriftDetector(
        reference_data,
        threshold=0.05
    )

    print("\nTest 1: No Drift Expected")

    current_no_drift = pd.DataFrame({
        "temperature": np.random.normal(75, 15, 500),
        "vibration": np.random.normal(0.5, 0.2, 500),
        "pressure": np.random.normal(100, 20, 500)
    })

    drift_detected, results = detector.check_all_features(
        current_no_drift
    )

    print(f"Drift detected: {drift_detected}")

    for feature, result in results.items():

        status = (
            " DRIFT"
            if result["drift_detected"]
            else " OK"
        )

        print(
            f"{feature}: {status} "
            f"(p={result['p_value']:.4f})"
        )

    print("\nTest 2: Drift Expected")

    current_drift = pd.DataFrame({
        "temperature": np.random.normal(95, 15, 500),
        "vibration": np.random.normal(0.8, 0.2, 500),
        "pressure": np.random.normal(100, 20, 500)
    })

    drift_detected, results = detector.check_all_features(
        current_drift
    )

    print(f"Drift detected: {drift_detected}")

    for feature, result in results.items():

        status = (
            " DRIFT"
            if result["drift_detected"]
            else " OK"
        )

        print(
            f"{feature}: {status} "
            f"(p={result['p_value']:.4f})"
        )

    detector.save_report()

    print("\n" + "=" * 60)
    print(" TEST COMPLETE")
    print("=" * 60)