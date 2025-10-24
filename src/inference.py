"""
Production Inference System for Predictive Maintenance
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictiveMaintenanceInference:
    def __init__(self, model_path: str, preprocessor_path: str, 
                 feature_engineer_path: str, feature_columns_path: str):
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_engineer = joblib.load(feature_engineer_path)
            
            with open(feature_columns_path, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            
            logger.info("Inference system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            raise
    
    def predict(self, new_data: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        try:
            if 'unit_id' not in new_data.columns:
                logger.warning("unit_id column not found. Creating sequential unit_id.")
                new_data = new_data.copy()
                new_data['unit_id'] = range(1, len(new_data) + 1)
            
            if 'time_cycles' not in new_data.columns:
                logger.warning("time_cycles column not found. Creating sequential time_cycles.")
                new_data['time_cycles'] = range(1, len(new_data) + 1)
            
            processed_data = self.preprocessor.remove_constant_sensors(new_data.copy())
            available_features = [col for col in self.preprocessor.feature_columns 
                                if col in processed_data.columns]
            processed_data[available_features] = self.preprocessor.scaler.transform(
                processed_data[available_features])
            
            engineered_data = self.feature_engineer.engineer_features(processed_data)
            
            for col in self.feature_columns:
                if col not in engineered_data.columns:
                    engineered_data[col] = 0
                    logger.warning(f"Added missing feature {col} with default value")
            
            predictions = self.model.predict(engineered_data[self.feature_columns])
            alerts = self._generate_alerts(predictions)
            
            logger.info(f"Successfully generated predictions for {len(predictions)} records")
            
            return predictions, alerts, engineered_data
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _generate_alerts(self, predictions: np.ndarray, 
                        critical_threshold: int = 10, 
                        warning_threshold: int = 30) -> List[str]:
        alerts = []
        for i, rul in enumerate(predictions):
            if rul <= critical_threshold:
                alert_msg = f"CRITICAL: Engine {i+1} has {rul:.1f} cycles remaining - Immediate maintenance required!"
            elif rul <= warning_threshold:
                alert_msg = f"WARNING: Engine {i+1} has {rul:.1f} cycles remaining - Schedule maintenance soon."
            else:
                alert_msg = f"NORMAL: Engine {i+1} has {rul:.1f} cycles remaining - No immediate action needed."
            alerts.append(alert_msg)
        
        return alerts

def main():
    parser = argparse.ArgumentParser(description='Predictive Maintenance Inference System')
    parser.add_argument('--input', required=True, help='Input sensor data file (CSV format)')
    parser.add_argument('--output', required=True, help='Output predictions file (CSV format)')
    parser.add_argument('--model', default='models/predictive_maintenance_model.pkl', 
                       help='Path to trained model file')
    parser.add_argument('--preprocessor', default='models/preprocessor.pkl',
                       help='Path to preprocessing pipeline')
    parser.add_argument('--feature_engineer', default='models/feature_engineer.pkl',
                       help='Path to feature engineering pipeline')
    parser.add_argument('--feature_columns', default='models/feature_columns.txt',
                       help='Path to feature columns list')
    
    args = parser.parse_args()
    
    try:
        inference_system = PredictiveMaintenanceInference(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            feature_engineer_path=args.feature_engineer,
            feature_columns_path=args.feature_columns
        )
        
        input_data = pd.read_csv(args.input)
        predictions, alerts, processed_data = inference_system.predict(input_data)
        
        output_df = input_data.copy()
        output_df['predicted_rul'] = predictions
        output_df['maintenance_alert'] = alerts
        output_df.to_csv(args.output, index=False)
        
        print(f"Successfully processed {len(predictions)} records")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()