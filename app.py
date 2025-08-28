import streamlit as st
import numpy as np
from cml_model import QuantumDeliveryMLModel
from quantum import QuantumInspiredPathFinder, create_demo_city

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .big-title {font-size: 2.5rem; font-weight: bold; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
    .city-label {font-size: 1.1rem; color: #7f8c8d;}
    .section-title {color: #2c3e50; font-size: 1.3rem; margin-top: 1.5em;}
    .gradient-btn {background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; padding: 0.8em 2em; border-radius: 10px; font-size: 1.1rem; margin-bottom: 1em;}
    .legend-quantum {background: linear-gradient(90deg, #667eea, #764ba2); width: 20px; height: 4px; border-radius: 2px; display: inline-block;}
    .legend-classical {background: linear-gradient(90deg, #e74c3c, #f39c12); width: 20px; height: 4px; border-radius: 2px; display: inline-block;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚀 Quantum Path Planning for Delivery Vehicles</div>', unsafe_allow_html=True)
st.markdown('<div class="city-label">Fleet Optimization using Machine Learning & Quantum-Inspired Algorithms</div>', unsafe_allow_html=True)

nodes, edges = create_demo_city()
node_labels = {
    "A": "Warehouse",
    "B": "Mall",
    "C": "Downtown",
    "D": "Business Park",
    "E": "Highway Junction",
    "F": "Residential",
    "G": "Suburb",
    "H": "Airport"
}

st.markdown('<div class="section-title">📍 Select Delivery Route</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox("Pick-up Location", options=list(nodes), format_func=lambda x: f"{x} ({node_labels[x]})")
with col2:
    end = st.selectbox("Delivery Destination", options=list(nodes), format_func=lambda x: f"{x} ({node_labels[x]})")

place_order = st.button("📦 Place Order", key="place_order_btn")

if 'order_placed' not in st.session_state:
    st.session_state.order_placed = False
if 'delivery_type' not in st.session_state:
    st.session_state.delivery_type = None

if place_order:
    if start == end:
        st.error("Pickup and delivery locations must be different!")
    else:
        st.session_state.order_placed = True
        st.session_state.start = start
        st.session_state.end = end
        st.session_state.delivery_type = None

if st.session_state.order_placed:
    st.markdown('<div class="section-title">🚚 Delivery Type</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🚨 Emergency Delivery", key="emergency_btn"):
            st.session_state.delivery_type = "emergency"
    with col4:
        if st.button("📦 Normal Delivery", key="normal_btn"):
            st.session_state.delivery_type = "normal"

    if st.session_state.delivery_type:
        st.markdown('<div class="section-title">🌦️ Current Conditions</div>', unsafe_allow_html=True)
        constraint_cols = st.columns(2)
        with constraint_cols[0]:
            weather = st.radio("Weather", ["sunny", "rainy", "foggy", "cloudy"], horizontal=True)
        with constraint_cols[1]:
            time_of_day = st.radio("Time Period", ["morning", "afternoon", "evening", "night"], horizontal=True)

        optimize = st.button("🧠 Find Optimal Path", key="optimize_btn")
        if optimize:
            # ML & Quantum logic
            st.info("Optimizing delivery route using Quantum Algorithm...")
            ml_model = QuantumDeliveryMLModel()
            ml_model.train_models()
            quantum_finder = QuantumInspiredPathFinder(ml_model)
            quantum_finder.create_city_graph(nodes, edges)

            constraints = {
                "weather": weather,
                "time_period": time_of_day,
                "delivery_type": st.session_state.delivery_type,
                "vehicle_capacity": 100,
                "order_weight": 10  # Or randomize/let user select
            }
            results = quantum_finder.compare_algorithms(st.session_state.start, st.session_state.end, constraints)

            st.markdown('<div class="section-title">🗺️ City Navigation Map</div>', unsafe_allow_html=True)
            st.caption("Below: Quantum and Traditional/GPS routes overlayed (graphical visualization not implemented in this basic example).")
            st.markdown(
                '<span class="legend-quantum"></span> Quantum Route &nbsp;&nbsp;'
                '<span class="legend-classical"></span> Traditional GPS', unsafe_allow_html=True
            )

            st.markdown('<div class="section-title">📊 Path Comparison Results</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("### 🌟 Quantum Algorithm")
                st.write(f"**Path:** {' → '.join(results['quantum']['path']) if results['quantum']['path'] else '-'}")
                st.write(f"**Distance:** {results['quantum']['distance']:.2f} km")
                st.write(f"**Time:** {results['quantum']['estimated_delivery_time']:.1f} min")
                st.write(f"**Fuel:** {results['quantum']['estimated_fuel_consumption']:.2f} L")
                st.write(f"**CO2:** {results['quantum']['estimated_co2_emission']:.2f} kg")
            with cols[1]:
                st.markdown("### 🔍 Traditional GPS")
                st.write(f"**Path:** {' → '.join(results['classical']['path']) if results['classical']['path'] else '-'}")
                st.write(f"**Distance:** {results['classical']['distance']:.2f} km")
                st.write(f"**Time:** {results['classical']['estimated_delivery_time']:.1f} min")
                st.write(f"**Fuel:** {results['classical']['estimated_fuel_consumption']:.2f} L")
                st.write(f"**CO2:** {results['classical']['estimated_co2_emission']:.2f} kg")

            if "improvements" in results:
                st.markdown('<div class="section-title">🎯 Quantum Optimization Benefits</div>', unsafe_allow_html=True)
                savings = results["improvements"]
                savings_cols = st.columns(3)
                savings_cols[0].metric("Time Saved", f"{savings['time_saved_minutes']:.1f} min", f"{savings['time_saved_percentage']:.1f}%")
                savings_cols[1].metric("Fuel Saved", f"{savings['fuel_saved_liters']:.2f} L", f"{savings['fuel_saved_percentage']:.1f}%")
                savings_cols[2].metric("CO2 Reduced", f"{savings['co2_saved_kg']:.2f} kg", f"{savings['co2_saved_percentage']:.1f}%")
