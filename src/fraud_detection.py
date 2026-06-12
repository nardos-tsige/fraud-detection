"""
Fraud Detection Module - Reusable functions with error handling
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any

class FraudDetector:
    """Main fraud detection class with proper error handling"""
    
    def __init__(self, model_path: str = 'models/best_fraud_model.pkl', 
                 scaler_path: str = 'models/scaler.pkl'):
        """
        Initialize the fraud detector with trained model and scaler
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the fitted scaler file
        
        Raises:
            FileNotFoundError: If model or scaler files don't exist
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded successfully from {model_path}")
            
        except FileNotFoundError as e:
            print(f"File error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading model: {e}")
            raise
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess raw features for prediction
        
        Args:
            df: DataFrame with raw transaction features
        
        Returns:
            Scaled feature array ready for prediction
        
        Raises:
            ValueError: If required columns are missing
        """
        required_features = ['purchase_value', 'age', 'hour_of_day', 
                            'day_of_week', 'time_since_signup_hours', 
                            'new_user_1h', 'users_per_device']
        
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X = df[required_features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict fraud for given features"""
        if features is None:
            raise ValueError("Features cannot be None")
        if len(features) == 0:
            raise ValueError("Features array is empty")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get probability scores for predictions"""
        if features is None:
            raise ValueError("Features cannot be None")
        
        return self.model.predict_proba(features)
    
    def predict_single_transaction(self, purchase_value: float, age: int, 
                                   hour_of_day: int, day_of_week: int,
                                   time_since_signup_hours: float,
                                   new_user_1h: int, users_per_device: int) -> Dict:
        """Predict fraud for a single transaction"""
        features = np.array([[purchase_value, age, hour_of_day, day_of_week,
                              time_since_signup_hours, new_user_1h, users_per_device]])
        
        features_scaled = self.scaler.transform(features)
        prediction = int(self.predict(features_scaled)[0])
        probability = float(self.predict_proba(features_scaled)[0][1])
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load CSV data with error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError(f"File is empty: {filepath}")
        
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
        
    except FileNotFoundError as e:
        print(f"{e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        raise


if __name__ == "__main__":
    print("Testing Fraud Detection Module")
    
    try:
        detector = FraudDetector()
        
        result = detector.predict_single_transaction(
            purchase_value=250.00, age=30, hour_of_day=14, day_of_week=2,
            time_since_signup_hours=0.5, new_user_1h=1, users_per_device=1
        )
        
        print(f"Test Prediction: {result}")
        print("Module working correctly!")
        
    except Exception as e:
        print(f"Module test failed: {e}")