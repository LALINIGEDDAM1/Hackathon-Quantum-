import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class QuantumDeliveryMLModel:
    def __init__(self):
        """
        ML Model for Quantum Path Planning in Delivery Systems
        
        This model predicts:
        1. Traffic weights for road segments
        2. Delivery time estimates
        3. Fuel consumption predictions
        4. Weather impact on routes
        """
        
        # Initialize models for different predictions
        self.traffic_predictor = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        
        self.delivery_time_predictor = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1
        )
        
        self.fuel_consumption_predictor = RandomForestRegressor(
            n_estimators=80,
            random_state=42
        )
        
        # Scalers for feature normalization
        self.traffic_scaler = StandardScaler()
        self.delivery_scaler = StandardScaler()
        self.fuel_scaler = StandardScaler()
        
        # Label encoders for categorical variables
        self.weather_encoder = LabelEncoder()
        self.road_type_encoder = LabelEncoder()
        self.time_period_encoder = LabelEncoder()
        
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic training data for the delivery system
        This simulates historical delivery data
        """
        np.random.seed(42)
        
        # Define possible values for categorical features
        weather_conditions = ['sunny', 'rainy', 'cloudy', 'foggy']
        road_types = ['highway', 'city_road', 'residential', 'commercial']
        time_periods = ['morning', 'afternoon', 'evening', 'night']
        delivery_types = ['normal', 'emergency']
        
        data = []
        
        for _ in range(n_samples):
            # Random feature generation
            weather = np.random.choice(weather_conditions)
            road_type = np.random.choice(road_types)
            time_period = np.random.choice(time_periods)
            delivery_type = np.random.choice(delivery_types)
            
            # Numerical features
            distance = np.random.uniform(0.5, 20.0)  # km
            base_traffic = np.random.uniform(0.1, 1.0)
            vehicle_capacity = np.random.choice([50, 100, 150, 200])  # kg
            order_weight = np.random.uniform(1, vehicle_capacity * 0.8)
            
            # Calculate traffic weight based on conditions
            traffic_multiplier = 1.0
            
            # Weather impact
            if weather == 'rainy':
                traffic_multiplier *= np.random.uniform(1.3, 1.8)
            elif weather == 'foggy':
                traffic_multiplier *= np.random.uniform(1.2, 1.5)
            
            # Time period impact
            if time_period == 'morning' or time_period == 'evening':
                traffic_multiplier *= np.random.uniform(1.2, 1.6)
            elif time_period == 'afternoon':
                traffic_multiplier *= np.random.uniform(1.1, 1.3)
            
            # Road type impact
            if road_type == 'city_road':
                traffic_multiplier *= np.random.uniform(1.1, 1.4)
            elif road_type == 'residential':
                traffic_multiplier *= np.random.uniform(0.8, 1.1)
            elif road_type == 'highway':
                traffic_multiplier *= np.random.uniform(0.9, 1.2)
            
            final_traffic_weight = base_traffic * traffic_multiplier
            final_traffic_weight = min(final_traffic_weight, 2.0)  # Cap at 2.0
            
            # Calculate delivery time (minutes)
            base_time = distance * 3  # 3 minutes per km base
            delivery_time = base_time * final_traffic_weight
            
            if delivery_type == 'emergency':
                delivery_time *= 0.8  # Emergency deliveries are faster
            
            # Calculate fuel consumption (liters)
            base_fuel = distance * 0.1  # 0.1L per km base
            fuel_consumption = base_fuel * (1 + (final_traffic_weight - 1) * 0.5)
            fuel_consumption += (order_weight / vehicle_capacity) * base_fuel * 0.2
            
            data.append({
                'weather': weather,
                'road_type': road_type,
                'time_period': time_period,
                'delivery_type': delivery_type,
                'distance': distance,
                'base_traffic': base_traffic,
                'vehicle_capacity': vehicle_capacity,
                'order_weight': order_weight,
                'traffic_weight': final_traffic_weight,
                'delivery_time': delivery_time,
                'fuel_consumption': fuel_consumption
            })
        
        return pd.DataFrame(data)
    def prepare_features(self, df, fit_encoders=False):
        """
        Prepare features for model training/prediction
        """
        df = df.copy()

        # Known categories for fallback
        known_weather = list(getattr(self.weather_encoder, "classes_", ["sunny", "rainy", "cloudy", "foggy"]))
        known_road_type = list(getattr(self.road_type_encoder, "classes_", ["highway", "city_road", "residential", "commercial"]))
        known_time_period = list(getattr(self.time_period_encoder, "classes_", ["morning", "afternoon", "evening", "night"]))

        # Replace unknown categories with a default known value
        df['weather'] = df['weather'].apply(lambda x: x if x in known_weather else "sunny")
        df['road_type'] = df['road_type'].apply(lambda x: x if x in known_road_type else "city_road")
        df['time_period'] = df['time_period'].apply(lambda x: x if x in known_time_period else "afternoon")

        # Encode categorical variables
        if fit_encoders:
            self.weather_encoder.fit(df['weather'])
            self.road_type_encoder.fit(df['road_type'])
            self.time_period_encoder.fit(df['time_period'])

        df['weather_encoded'] = self.weather_encoder.transform(df['weather'])
        df['road_type_encoded'] = self.road_type_encoder.transform(df['road_type'])
        df['time_period_encoded'] = self.time_period_encoder.transform(df['time_period'])

        # Binary feature for emergency delivery
        df['is_emergency'] = (df['delivery_type'] == 'emergency').astype(int)

        # Interaction features
        df['weight_ratio'] = df['order_weight'] / df['vehicle_capacity']
        df['traffic_distance_interaction'] = df['base_traffic'] * df['distance']

        return df
    

    
    def train_models(self, df=None):
        """
        Train all ML models
        """
        if df is None:
            print("Generating synthetic training data...")
            df = self.generate_synthetic_data()
        
        print(f"Training on {len(df)} samples...")
        
        # Prepare features
        df_processed = self.prepare_features(df, fit_encoders=True)
        
        # Define feature columns
        feature_cols = [
            'weather_encoded', 'road_type_encoded', 'time_period_encoded',
            'is_emergency', 'distance', 'base_traffic', 'vehicle_capacity',
            'order_weight', 'weight_ratio', 'traffic_distance_interaction'
        ]
        
        X = df_processed[feature_cols]
        
        # Train traffic weight predictor
        print("Training traffic weight predictor...")
        y_traffic = df_processed['traffic_weight']
        X_train_traffic, X_test_traffic, y_train_traffic, y_test_traffic = train_test_split(
            X, y_traffic, test_size=0.2, random_state=42
        )
        
        X_train_traffic_scaled = self.traffic_scaler.fit_transform(X_train_traffic)
        X_test_traffic_scaled = self.traffic_scaler.transform(X_test_traffic)
        
        self.traffic_predictor.fit(X_train_traffic_scaled, y_train_traffic)
        traffic_pred = self.traffic_predictor.predict(X_test_traffic_scaled)
        print(f"Traffic Weight Predictor - MAE: {mean_absolute_error(y_test_traffic, traffic_pred):.3f}, R2: {r2_score(y_test_traffic, traffic_pred):.3f}")
        
        # Train delivery time predictor
        print("Training delivery time predictor...")
        y_delivery = df_processed['delivery_time']
        X_train_del, X_test_del, y_train_del, y_test_del = train_test_split(
            X, y_delivery, test_size=0.2, random_state=42
        )
        
        X_train_del_scaled = self.delivery_scaler.fit_transform(X_train_del)
        X_test_del_scaled = self.delivery_scaler.transform(X_test_del)
        
        self.delivery_time_predictor.fit(X_train_del_scaled, y_train_del)
        delivery_pred = self.delivery_time_predictor.predict(X_test_del_scaled)
        print(f"Delivery Time Predictor - MAE: {mean_absolute_error(y_test_del, delivery_pred):.3f} min, R2: {r2_score(y_test_del, delivery_pred):.3f}")
        
        # Train fuel consumption predictor
        print("Training fuel consumption predictor...")
        y_fuel = df_processed['fuel_consumption']
        X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(
            X, y_fuel, test_size=0.2, random_state=42
        )
        
        X_train_fuel_scaled = self.fuel_scaler.fit_transform(X_train_fuel)
        X_test_fuel_scaled = self.fuel_scaler.transform(X_test_fuel)
        
        self.fuel_consumption_predictor.fit(X_train_fuel_scaled, y_train_fuel)
        fuel_pred = self.fuel_consumption_predictor.predict(X_test_fuel_scaled)
        print(f"Fuel Consumption Predictor - MAE: {mean_absolute_error(y_test_fuel, fuel_pred):.3f} L, R2: {r2_score(y_test_fuel, fuel_pred):.3f}")
        
        self.is_trained = True
        print("All models trained successfully!")
        
        return {
            'traffic_mae': mean_absolute_error(y_test_traffic, traffic_pred),
            'delivery_mae': mean_absolute_error(y_test_del, delivery_pred),
            'fuel_mae': mean_absolute_error(y_test_fuel, fuel_pred)
        }
    
    def predict_traffic_weights(self, route_segments):
        """
        Predict traffic weights for route segments
        
        Args:
            route_segments: List of dictionaries with segment information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Convert to DataFrame
        df = pd.DataFrame(route_segments)
        df_processed = self.prepare_features(df, fit_encoders=False)
        
        feature_cols = [
            'weather_encoded', 'road_type_encoded', 'time_period_encoded',
            'is_emergency', 'distance', 'base_traffic', 'vehicle_capacity',
            'order_weight', 'weight_ratio', 'traffic_distance_interaction'
        ]
        
        X = df_processed[feature_cols]
        X_scaled = self.traffic_scaler.transform(X)
        
        predictions = self.traffic_predictor.predict(X_scaled)
        return predictions
    
    def predict_delivery_metrics(self, delivery_info):
        """
        Predict delivery time and fuel consumption
        
        Args:
            delivery_info: Dictionary with delivery information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        df = pd.DataFrame([delivery_info])
        df_processed = self.prepare_features(df, fit_encoders=False)
        
        feature_cols = [
            'weather_encoded', 'road_type_encoded', 'time_period_encoded',
            'is_emergency', 'distance', 'base_traffic', 'vehicle_capacity',
            'order_weight', 'weight_ratio', 'traffic_distance_interaction'
        ]
        
        X = df_processed[feature_cols]
        
        # Predict delivery time
        X_delivery_scaled = self.delivery_scaler.transform(X)
        delivery_time = self.delivery_time_predictor.predict(X_delivery_scaled)[0]
        
        # Predict fuel consumption
        X_fuel_scaled = self.fuel_scaler.transform(X)
        fuel_consumption = self.fuel_consumption_predictor.predict(X_fuel_scaled)[0]
        
        return {
            'estimated_delivery_time': delivery_time,
            'estimated_fuel_consumption': fuel_consumption,
            'estimated_co2_emission': fuel_consumption * 2.31  # kg CO2 per liter of fuel
        }
    
    def get_quantum_algorithm_input(self, city_graph, constraints):
        """
        Generate optimized edge weights for quantum algorithm
        
        Args:
            city_graph: Graph representation of the city
            constraints: Current constraints (weather, time, delivery type, etc.)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        optimized_weights = {}
        
        for edge in city_graph['edges']:
            # Create segment info for prediction
            segment_info = {
                'weather': constraints.get('weather', 'sunny'),
                'road_type': edge.get('road_type', 'city_road'),
                'time_period': constraints.get('time_period', 'afternoon'),
                'delivery_type': constraints.get('delivery_type', 'normal'),
                'distance': edge['distance'],
                'base_traffic': edge.get('base_traffic', 0.5),
                'vehicle_capacity': constraints.get('vehicle_capacity', 100),
                'order_weight': constraints.get('order_weight', 5)
            }
            
            # Predict traffic weight
            predicted_weight = self.predict_traffic_weights([segment_info])[0]
            
            edge_key = f"{edge['from']}_{edge['to']}"
            optimized_weights[edge_key] = predicted_weight
        
        return optimized_weights
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'traffic_predictor': self.traffic_predictor,
            'delivery_time_predictor': self.delivery_time_predictor,
            'fuel_consumption_predictor': self.fuel_consumption_predictor,
            'traffic_scaler': self.traffic_scaler,
            'delivery_scaler': self.delivery_scaler,
            'fuel_scaler': self.fuel_scaler,
            'weather_encoder': self.weather_encoder,
            'road_type_encoder': self.road_type_encoder,
            'time_period_encoder': self.time_period_encoder,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.traffic_predictor = model_data['traffic_predictor']
        self.delivery_time_predictor = model_data['delivery_time_predictor']
        self.fuel_consumption_predictor = model_data['fuel_consumption_predictor']
        self.traffic_scaler = model_data['traffic_scaler']
        self.delivery_scaler = model_data['delivery_scaler']
        self.fuel_scaler = model_data['fuel_scaler']
        self.weather_encoder = model_data['weather_encoder']
        self.road_type_encoder = model_data['road_type_encoder']
        self.time_period_encoder = model_data['time_period_encoder']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")

# Example usage and integration
def integrate_ml_with_quantum_system():
    """
    Example of how to integrate ML model with quantum path planning
    """
    
    # Initialize and train the ML model
    ml_model = QuantumDeliveryMLModel()
    training_metrics = ml_model.train_models()
    
    print("\n=== ML Model Integration Example ===")
    
    # Example city graph structure
    city_graph = {
        'nodes': ['A', 'B', 'C', 'D', 'E'],
        'edges': [
            {'from': 'A', 'to': 'B', 'distance': 5.0, 'road_type': 'highway', 'base_traffic': 0.3},
            {'from': 'A', 'to': 'C', 'distance': 3.0, 'road_type': 'city_road', 'base_traffic': 0.6},
            {'from': 'B', 'to': 'D', 'distance': 4.0, 'road_type': 'residential', 'base_traffic': 0.2},
            {'from': 'C', 'to': 'D', 'distance': 6.0, 'road_type': 'commercial', 'base_traffic': 0.8},
            {'from': 'D', 'to': 'E', 'distance': 2.0, 'road_type': 'city_road', 'base_traffic': 0.4}
        ]
    }
    
    # Example constraints (user selections)
    constraints = {
        'weather': 'rainy',
        'time_period': 'evening',
        'delivery_type': 'emergency',
        'vehicle_capacity': 100,
        'order_weight': 15
    }
    
    print(f"Current constraints: {constraints}")
    
    # Get optimized weights for quantum algorithm
    quantum_weights = ml_model.get_quantum_algorithm_input(city_graph, constraints)
    print(f"\nOptimized edge weights for quantum algorithm:")
    for edge, weight in quantum_weights.items():
        print(f"  {edge}: {weight:.3f}")
    
    # Predict delivery metrics for the route
    total_distance = sum(edge['distance'] for edge in city_graph['edges'])
    avg_traffic = np.mean(list(quantum_weights.values()))
    
    delivery_info = {
    'weather': constraints['weather'],
    'road_type': 'city_road',  # <-- Use a valid category
    'time_period': constraints['time_period'],
    'delivery_type': constraints['delivery_type'],
    'distance': total_distance,
    'base_traffic': avg_traffic,
    'vehicle_capacity': constraints['vehicle_capacity'],
    'order_weight': constraints['order_weight']
}

    metrics = ml_model.predict_delivery_metrics(delivery_info)
    print(f"\nPredicted delivery metrics:")
    print(f"  Estimated delivery time: {metrics['estimated_delivery_time']:.1f} minutes")
    print(f"  Estimated fuel consumption: {metrics['estimated_fuel_consumption']:.2f} liters")
    print(f"  Estimated CO2 emission: {metrics['estimated_co2_emission']:.2f} kg")
    
    return ml_model, quantum_weights, metrics

if __name__ == "__main__":
    # Run the integration example
    model, weights, predictions = integrate_ml_with_quantum_system()
    
    print("\n=== ML Model Ready for Integration ===")
    print("The ML model can now provide intelligent input to your quantum algorithm!")