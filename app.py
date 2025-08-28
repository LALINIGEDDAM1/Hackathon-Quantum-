import streamlit as st
import pandas as pd
from cml_model import QuantumDeliveryMLModel
from quantum import QuantumInspiredPathFinder, create_demo_city, visualize_city_graph

st.set_page_config(page_title="Quantum Delivery Optimization", layout="wide")
st.title("🚚 Quantum Delivery Optimization Demo")

# Sidebar for scenario selection and constraints
st.sidebar.header("Scenario & Constraints")

# Scenario selection
scenarios = {
    "Sunny Morning - Normal Delivery": {
        "weather": "sunny",
        "time_period": "morning",
        "delivery_type": "normal",
        "vehicle_capacity": 100,
        "order_weight": 8,
        "start": "A",
        "end": "H"
    },
    "Rainy Evening - Emergency Delivery": {
        "weather": "rainy",
        "time_period": "evening",
        "delivery_type": "emergency",
        "vehicle_capacity": 100,
        "order_weight": 3,
        "start": "A",
        "end": "H"
    },
    "Foggy Afternoon - Normal Delivery": {
        "weather": "foggy",
        "time_period": "afternoon",
        "delivery_type": "normal",
        "vehicle_capacity": 150,
        "order_weight": 12,
        "start": "B",
        "end": "G"
    },
    "Custom": {}
}

selected_scenario = st.sidebar.selectbox("Choose a Scenario", list(scenarios.keys()))

# Load scenario values or let user pick
if selected_scenario != "Custom":
    scenario = scenarios[selected_scenario]
    weather = scenario["weather"]
    time_period = scenario["time_period"]
    delivery_type = scenario["delivery_type"]
    vehicle_capacity = scenario["vehicle_capacity"]
    order_weight = scenario["order_weight"]
    start = scenario["start"]
    end = scenario["end"]
else:
    weather = st.sidebar.selectbox("Weather", ["sunny", "rainy", "cloudy", "foggy"])
    time_period = st.sidebar.selectbox("Time Period", ["morning", "afternoon", "evening", "night"])
    delivery_type = st.sidebar.selectbox("Delivery Type", ["normal", "emergency"])
    vehicle_capacity = st.sidebar.slider("Vehicle Capacity (kg)", 50, 200, 100, step=10)
    order_weight = st.sidebar.slider("Order Weight (kg)", 1, vehicle_capacity, 5)
    start = st.sidebar.text_input("Start Node", "A")
    end = st.sidebar.text_input("End Node", "H")

constraints = {
    "weather": weather,
    "time_period": time_period,
    "delivery_type": delivery_type,
    "vehicle_capacity": vehicle_capacity,
    "order_weight": order_weight
}

# Main app area
st.header("Step 1: Train Machine Learning Model")
with st.spinner("Training ML model..."):
    ml_model = QuantumDeliveryMLModel()
    metrics = ml_model.train_models()
st.success("ML model trained!")

st.write("**ML Model Metrics:**")
st.json(metrics)

st.header("Step 2: Create Demo City Graph")
nodes, edges = create_demo_city()
quantum_finder = QuantumInspiredPathFinder(ml_model)
quantum_finder.create_city_graph(nodes, edges)
st.write(f"City graph created with {len(nodes)} nodes and {len(edges)} edges.")

# Visualization
st.subheader("City Graph Visualization")
if st.button("Show City Graph"):
    visualize_city_graph(quantum_finder)

st.header("Step 3: Run Delivery Optimization")
st.write(f"**Route:** {start} → {end}")
st.write("**Constraints:**")
st.json(constraints)

if st.button("Run Quantum vs Classical Optimization"):
    results = quantum_finder.compare_algorithms(start, end, constraints)
    st.subheader("Detailed Results")
    if results["quantum"]["path"]:
        st.markdown("### 🌟 Quantum Algorithm")
        st.write(f"Path: {' → '.join(results['quantum']['path'])}")
        st.write(f"Total Distance: {results['quantum']['distance']:.2f} km")
        st.write(f"Delivery Time: {results['quantum']['estimated_delivery_time']:.1f} minutes")
        st.write(f"Fuel Consumption: {results['quantum']['estimated_fuel_consumption']:.2f} liters")
        st.write(f"CO2 Emission: {results['quantum']['estimated_co2_emission']:.2f} kg")
    if results["classical"]["path"]:
        st.markdown("### 🔍 Classical Dijkstra Algorithm")
        st.write(f"Path: {' → '.join(results['classical']['path'])}")
        st.write(f"Total Distance: {results['classical']['distance']:.2f} km")
        st.write(f"Delivery Time: {results['classical']['estimated_delivery_time']:.1f} minutes")
        st.write(f"Fuel Consumption: {results['classical']['estimated_fuel_consumption']:.2f} liters")
        st.write(f"CO2 Emission: {results['classical']['estimated_co2_emission']:.2f} kg")
    if "improvements" in results:
        st.markdown("### 🎯 Quantum Improvements")
        st.write(f"Time Saved: {results['improvements']['time_saved_minutes']:.1f} minutes ({results['improvements']['time_saved_percentage']:.1f}%)")
        st.write(f"Fuel Saved: {results['improvements']['fuel_saved_liters']:.2f} liters ({results['improvements']['fuel_saved_percentage']:.1f}%)")
        st.write(f"CO2 Reduced: {results['improvements']['co2_saved_kg']:.2f} kg ({results['improvements']['co2_saved_percentage']:.1f}%)")
else:
    st.info("Press 'Run Quantum vs Classical Optimization' to compare algorithms.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and NetworkX.")
