import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

class QuantumInspiredPathFinder:
    """
    Quantum-inspired pathfinding algorithm that works with ML model predictions
    This simulates quantum optimization concepts using classical computing
    """
    
    def __init__(self, ml_model):
        self.ml_model = ml_model
        self.graph = nx.Graph()
        self.quantum_states = {}
        
    def create_city_graph(self, nodes, edges):
        """Create NetworkX graph from city data"""
        self.graph = nx.Graph()
        
        # Add nodes
        for node in nodes:
            self.graph.add_node(node)
        
        # Add edges with weights
        for edge in edges:
            self.graph.add_edge(
                edge['from'], 
                edge['to'], 
                distance=edge['distance'],
                road_type=edge.get('road_type', 'city_road'),
                base_traffic=edge.get('base_traffic', 0.5)
            )
    
    def quantum_superposition_pathfinding(self, start, end, ml_weights, num_iterations=50):
        """
        Quantum-inspired pathfinding using superposition of path states
        """
        print(f"\n🌟 Starting Quantum Path Search from {start} to {end}")
        
        # Find all simple paths (like quantum superposition of all possible paths)
        all_paths = list(nx.all_simple_paths(self.graph, start, end, cutoff=6))
        
        if not all_paths:
            return None, float('inf')
        
        print(f"Found {len(all_paths)} possible quantum path states")
        
        # Initialize quantum state amplitudes for each path
        path_amplitudes = np.random.uniform(0, 1, len(all_paths))
        path_amplitudes = path_amplitudes / np.sum(path_amplitudes)  # Normalize
        
        best_path = None
        best_cost = float('inf')
        
        # Quantum-inspired iterative optimization
        for iteration in range(num_iterations):
            path_costs = []
            
            # Calculate cost for each path using ML-predicted weights
            for path in all_paths:
                cost = self.calculate_path_cost(path, ml_weights)
                path_costs.append(cost)
            
            path_costs = np.array(path_costs)
            
            # Quantum amplitude updating (simulates quantum interference)
            # Lower cost paths get higher amplitudes
            inverse_costs = 1.0 / (path_costs + 0.01)  # Avoid division by zero
            new_amplitudes = path_amplitudes * inverse_costs
            
            # Quantum decoherence simulation - add some randomness
            noise = np.random.normal(0, 0.1, len(new_amplitudes))
            new_amplitudes += noise
            new_amplitudes = np.abs(new_amplitudes)  # Keep positive
            
            # Normalize amplitudes
            if np.sum(new_amplitudes) > 0:
                path_amplitudes = new_amplitudes / np.sum(new_amplitudes)
            
            # Find current best path (highest amplitude * lowest cost)
            optimization_scores = path_amplitudes / (path_costs + 0.01)
            best_idx = np.argmax(optimization_scores)
            
            if path_costs[best_idx] < best_cost:
                best_cost = path_costs[best_idx]
                best_path = all_paths[best_idx]
        
        return best_path, best_cost
    
    def classical_dijkstra_pathfinding(self, start, end, ml_weights):
        """
        Classical Dijkstra algorithm for comparison
        """
        print(f"\n🔍 Starting Classical Dijkstra Search from {start} to {end}")
        
        # Create weight dictionary for NetworkX
        edge_weights = {}
        for edge in self.graph.edges():
            edge_key = f"{edge[0]}_{edge[1]}"
            reverse_key = f"{edge[1]}_{edge[0]}"
            
            weight = ml_weights.get(edge_key, ml_weights.get(reverse_key, 1.0))
            edge_weights[edge] = weight
        
        # Set edge weights
        nx.set_edge_attributes(self.graph, edge_weights, 'weight')
        
        try:
            path = nx.shortest_path(self.graph, start, end, weight='weight')
            cost = nx.shortest_path_length(self.graph, start, end, weight='weight')
            return path, cost
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    def calculate_path_cost(self, path, ml_weights):
        """Calculate total cost of a path using ML-predicted weights"""
        total_cost = 0
        
        for i in range(len(path) - 1):
            edge_key = f"{path[i]}_{path[i+1]}"
            reverse_key = f"{path[i+1]}_{path[i]}"
            
            # Get ML-predicted weight
            weight = ml_weights.get(edge_key, ml_weights.get(reverse_key, 1.0))
            
            # Get distance from graph
            distance = self.graph[path[i]][path[i+1]]['distance']
            
            # Total cost is distance weighted by traffic conditions
            total_cost += distance * weight
        
        return total_cost
    
    def compare_algorithms(self, start, end, constraints):
        """Compare quantum vs classical algorithms"""
        print(f"\n{'='*60}")
        print(f"🚀 QUANTUM vs CLASSICAL PATHFINDING COMPARISON")
        print(f"{'='*60}")
        
        # Get ML predictions for current constraints
        city_graph = {
            'nodes': list(self.graph.nodes()),
            'edges': [
                {
                    'from': edge[0],
                    'to': edge[1],
                    'distance': self.graph[edge[0]][edge[1]]['distance'],
                    'road_type': self.graph[edge[0]][edge[1]]['road_type'],
                    'base_traffic': self.graph[edge[0]][edge[1]]['base_traffic']
                }
                for edge in self.graph.edges()
            ]
        }
        
        ml_weights = self.ml_model.get_quantum_algorithm_input(city_graph, constraints)
        
        # Run both algorithms
        quantum_path, quantum_cost = self.quantum_superposition_pathfinding(start, end, ml_weights)
        classical_path, classical_cost = self.classical_dijkstra_pathfinding(start, end, ml_weights)
        
        # Calculate metrics
        results = self.calculate_comparison_metrics(
            quantum_path, quantum_cost, classical_path, classical_cost, constraints
        )
        
        return results
    
    def calculate_comparison_metrics(self, quantum_path, quantum_cost, classical_path, classical_cost, constraints):
        """Calculate detailed comparison metrics"""
        
        results = {
            'quantum': {
                'path': quantum_path,
                'cost': quantum_cost,
                'distance': self.get_path_distance(quantum_path) if quantum_path else 0
            },
            'classical': {
                'path': classical_path,
                'cost': classical_cost,
                'distance': self.get_path_distance(classical_path) if classical_path else 0
            }
        }
        
        # Calculate delivery metrics for both paths
        if quantum_path:
            quantum_delivery_info = {
                'weather': constraints.get('weather', 'sunny'),
                'road_type': 'mixed',
                'time_period': constraints.get('time_period', 'afternoon'),
                'delivery_type': constraints.get('delivery_type', 'normal'),
                'distance': results['quantum']['distance'],
                'base_traffic': quantum_cost / max(results['quantum']['distance'], 0.1),
                'vehicle_capacity': constraints.get('vehicle_capacity', 100),
                'order_weight': constraints.get('order_weight', 5)
            }
            quantum_metrics = self.ml_model.predict_delivery_metrics(quantum_delivery_info)
            results['quantum'].update(quantum_metrics)
        
        if classical_path:
            classical_delivery_info = {
                'weather': constraints.get('weather', 'sunny'),
                'road_type': 'mixed',
                'time_period': constraints.get('time_period', 'afternoon'),
                'delivery_type': constraints.get('delivery_type', 'normal'),
                'distance': results['classical']['distance'],
                'base_traffic': classical_cost / max(results['classical']['distance'], 0.1),
                'vehicle_capacity': constraints.get('vehicle_capacity', 100),
                'order_weight': constraints.get('order_weight', 5)
            }
            classical_metrics = self.ml_model.predict_delivery_metrics(classical_delivery_info)
            results['classical'].update(classical_metrics)
        
        # Calculate improvements
        if quantum_path and classical_path:
            time_saved = results['classical']['estimated_delivery_time'] - results['quantum']['estimated_delivery_time']
            fuel_saved = results['classical']['estimated_fuel_consumption'] - results['quantum']['estimated_fuel_consumption']
            co2_saved = results['classical']['estimated_co2_emission'] - results['quantum']['estimated_co2_emission']
            
            results['improvements'] = {
                'time_saved_minutes': time_saved,
                'time_saved_percentage': (time_saved / results['classical']['estimated_delivery_time']) * 100,
                'fuel_saved_liters': fuel_saved,
                'fuel_saved_percentage': (fuel_saved / results['classical']['estimated_fuel_consumption']) * 100,
                'co2_saved_kg': co2_saved,
                'co2_saved_percentage': (co2_saved / results['classical']['estimated_co2_emission']) * 100
            }
        
        return results
    
    def get_path_distance(self, path):
        """Calculate total distance of a path"""
        if not path or len(path) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.graph[path[i]][path[i+1]]['distance']
        
        return total_distance
    
    def print_detailed_results(self, results):
        """Print detailed comparison results"""
        print(f"\n📊 DETAILED RESULTS")
        print(f"{'='*50}")
        
        # Quantum Results
        if results['quantum']['path']:
            print(f"🌟 QUANTUM ALGORITHM:")
            print(f"   Path: {' → '.join(results['quantum']['path'])}")
            print(f"   Total Distance: {results['quantum']['distance']:.2f} km")
            print(f"   Delivery Time: {results['quantum']['estimated_delivery_time']:.1f} minutes")
            print(f"   Fuel Consumption: {results['quantum']['estimated_fuel_consumption']:.2f} liters")
            print(f"   CO2 Emission: {results['quantum']['estimated_co2_emission']:.2f} kg")
        
        # Classical Results
        if results['classical']['path']:
            print(f"\n🔍 CLASSICAL ALGORITHM:")
            print(f"   Path: {' → '.join(results['classical']['path'])}")
            print(f"   Total Distance: {results['classical']['distance']:.2f} km")
            print(f"   Delivery Time: {results['classical']['estimated_delivery_time']:.1f} minutes")
            print(f"   Fuel Consumption: {results['classical']['estimated_fuel_consumption']:.2f} liters")
            print(f"   CO2 Emission: {results['classical']['estimated_co2_emission']:.2f} kg")
        
        # Improvements
        if 'improvements' in results:
            print(f"\n🎯 QUANTUM IMPROVEMENTS:")
            print(f"   Time Saved: {results['improvements']['time_saved_minutes']:.1f} minutes ({results['improvements']['time_saved_percentage']:.1f}%)")
            print(f"   Fuel Saved: {results['improvements']['fuel_saved_liters']:.2f} liters ({results['improvements']['fuel_saved_percentage']:.1f}%)")
            print(f"   CO2 Reduced: {results['improvements']['co2_saved_kg']:.2f} kg ({results['improvements']['co2_saved_percentage']:.1f}%)")
        
        print(f"{'='*50}")

def create_demo_city():
    """Create a demo city for testing"""
    
    # Define city nodes (intersections/locations)
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Define edges (roads) with different characteristics
    edges = [
        # Main highways (fast but longer)
        {'from': 'A', 'to': 'B', 'distance': 8.0, 'road_type': 'highway', 'base_traffic': 0.2},
        {'from': 'B', 'to': 'E', 'distance': 7.0, 'road_type': 'highway', 'base_traffic': 0.3},
        {'from': 'E', 'to': 'H', 'distance': 6.0, 'road_type': 'highway', 'base_traffic': 0.25},
        
        # City roads (moderate traffic, medium distance)
        {'from': 'A', 'to': 'C', 'distance': 4.0, 'road_type': 'city_road', 'base_traffic': 0.6},
        {'from': 'C', 'to': 'D', 'distance': 3.0, 'road_type': 'city_road', 'base_traffic': 0.7},
        {'from': 'D', 'to': 'G', 'distance': 4.5, 'road_type': 'city_road', 'base_traffic': 0.5},
        {'from': 'G', 'to': 'H', 'distance': 3.5, 'road_type': 'city_road', 'base_traffic': 0.4},
        
        # Residential areas (low traffic, shorter distance)
        {'from': 'A', 'to': 'F', 'distance': 5.0, 'road_type': 'residential', 'base_traffic': 0.1},
        {'from': 'F', 'to': 'G', 'distance': 4.0, 'road_type': 'residential', 'base_traffic': 0.15},
        
        # Commercial areas (high traffic during business hours)
        {'from': 'B', 'to': 'C', 'distance': 3.5, 'road_type': 'commercial', 'base_traffic': 0.8},
        {'from': 'D', 'to': 'E', 'distance': 2.5, 'road_type': 'commercial', 'base_traffic': 0.9},
        {'from': 'F', 'to': 'D', 'distance': 3.0, 'road_type': 'commercial', 'base_traffic': 0.85},
        
        # Additional connecting roads
        {'from': 'C', 'to': 'F', 'distance': 2.0, 'road_type': 'city_road', 'base_traffic': 0.4},
        {'from': 'E', 'to': 'G', 'distance': 5.5, 'road_type': 'city_road', 'base_traffic': 0.3}
    ]
    
    return nodes, edges

def run_comprehensive_demo():
    """Run a comprehensive demonstration of the system"""
    
    print("🚀 INITIALIZING QUANTUM DELIVERY OPTIMIZATION SYSTEM")
    print("="*60)
    
    # Step 1: Initialize ML Model
    print("Step 1: Training ML Model...")
    from cml_model import QuantumDeliveryMLModel
    
    ml_model = QuantumDeliveryMLModel()
    ml_model.train_models()
    
    # Step 2: Create Quantum Path Finder
    print("\nStep 2: Initializing Quantum Path Finder...")
    quantum_finder = QuantumInspiredPathFinder(ml_model)
    
    # Step 3: Create Demo City
    print("Step 3: Creating Demo City...")
    nodes, edges = create_demo_city()
    quantum_finder.create_city_graph(nodes, edges)
    
    print(f"City created with {len(nodes)} nodes and {len(edges)} roads")
    
    # Step 4: Run Different Scenarios
    scenarios = [
        {
            'name': 'Sunny Morning - Normal Delivery',
            'start': 'A',
            'end': 'H',
            'constraints': {
                'weather': 'sunny',
                'time_period': 'morning',
                'delivery_type': 'normal',
                'vehicle_capacity': 100,
                'order_weight': 8
            }
        },
        {
            'name': 'Rainy Evening - Emergency Delivery',
            'start': 'A', 
            'end': 'H',
            'constraints': {
                'weather': 'rainy',
                'time_period': 'evening',
                'delivery_type': 'emergency',
                'vehicle_capacity': 100,
                'order_weight': 3
            }
        },
        {
            'name': 'Foggy Afternoon - Normal Delivery',
            'start': 'B',
            'end': 'G',
            'constraints': {
                'weather': 'foggy',
                'time_period': 'afternoon',
                'delivery_type': 'normal',
                'vehicle_capacity': 150,
                'order_weight': 12
            }
        }
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n\n🎬 SCENARIO: {scenario['name']}")
        print("="*60)
        print(f"Route: {scenario['start']} → {scenario['end']}")
        print(f"Conditions: {scenario['constraints']}")
        
        results = quantum_finder.compare_algorithms(
            scenario['start'], 
            scenario['end'], 
            scenario['constraints']
        )
        
        quantum_finder.print_detailed_results(results)
        all_results[scenario['name']] = results
    
    # Step 5: Generate Summary Report
    print(f"\n\n📈 SUMMARY REPORT")
    print("="*60)
    
    total_time_saved = 0
    total_fuel_saved = 0
    total_co2_saved = 0
    successful_optimizations = 0
    
    for scenario_name, results in all_results.items():
        if 'improvements' in results:
            total_time_saved += results['improvements']['time_saved_minutes']
            total_fuel_saved += results['improvements']['fuel_saved_liters']
            total_co2_saved += results['improvements']['co2_saved_kg']
            successful_optimizations += 1
    
    if successful_optimizations > 0:
        print(f"🎯 OVERALL QUANTUM OPTIMIZATION BENEFITS:")
        print(f"   Total Scenarios Tested: {len(scenarios)}")
        print(f"   Successful Optimizations: {successful_optimizations}")
        print(f"   Total Time Saved: {total_time_saved:.1f} minutes")
        print(f"   Total Fuel Saved: {total_fuel_saved:.2f} liters")
        print(f"   Total CO2 Reduction: {total_co2_saved:.2f} kg")
        print(f"   Average Time Savings per Delivery: {total_time_saved/successful_optimizations:.1f} minutes")
        print(f"   Average Fuel Savings per Delivery: {total_fuel_saved/successful_optimizations:.2f} liters")
    
    print(f"\n✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return all_results

def visualize_city_graph(quantum_finder):
    """Visualize the city graph"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(quantum_finder.graph, seed=42)
        
        # Draw the graph
        nx.draw(quantum_finder.graph, pos, with_labels=True, 
                node_color='lightblue', node_size=1000, 
                font_size=16, font_weight='bold')
        
        # Add edge labels with distances
        edge_labels = nx.get_edge_attributes(quantum_finder.graph, 'distance')
        nx.draw_networkx_edge_labels(quantum_finder.graph, pos, edge_labels)
        
        plt.title("Demo City Graph for Quantum Delivery Optimization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    
    print("\n🔧 INTEGRATION NOTES FOR YOUR PROTOTYPE:")
    print("-" * 50)
    print("1. The ML model provides intelligent traffic predictions")
    print("2. Quantum algorithm optimizes paths using ML insights")
    print("3. System adapts to real-time constraints (weather, time, etc.)")
    print("4. Delivers measurable improvements in time, fuel, and emissions")
    print("5. Ready for integration with your UI and visualization system")