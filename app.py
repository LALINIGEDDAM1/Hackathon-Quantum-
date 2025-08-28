import streamlit as st
import pandas as pd
import networkx as nx
from cml_model import QuantumDeliveryMLModel
from quantum import QuantumInspiredPathFinder, create_demo_city

# --------- Streamlit Page Setup ---------
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

# --------- Demo City Setup ---------
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

G = nx.Graph()
for node in nodes:
    G.add_node(node)
for edge in edges:
    G.add_edge(edge['from'], edge['to'], distance=edge['distance'], road_type=edge['road_type'], base_traffic=edge['base_traffic'])

# --------- UI Controls ---------
st.markdown('<div class="section-title">📍 Select Delivery Route</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox("Pick-up Location", options=nodes, format_func=lambda x: f"{x} ({node_labels[x]})")
with col2:
    end = st.selectbox("Delivery Destination", options=nodes, format_func=lambda x: f"{x} ({node_labels[x]})")

if start == end:
    st.error("Pickup and delivery locations must be different!")
    st.stop()

st.markdown('<div class="section-title">🚚 Delivery Type</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    delivery_type = st.radio("Delivery Type", ["normal", "emergency"], horizontal=True)
with col4:
    order_weight = st.slider("Order Weight (kg)", 1, 50, 10)
vehicle_capacity = 100

st.markdown('<div class="section-title">🌦 Current Conditions</div>', unsafe_allow_html=True)
constraint_cols = st.columns(2)
with constraint_cols[0]:
    weather = st.radio("Weather", ["sunny", "rainy", "foggy", "cloudy"], horizontal=True)
with constraint_cols[1]:
    time_of_day = st.radio("Time Period", ["morning", "afternoon", "evening", "night"], horizontal=True)

# --------- Leaflet Map Visualization ---------
import folium
from streamlit_folium import st_folium

# Assign some demo lat/lon for each node (replace with real values if available)
node_coords = {
    "A": (17.387140, 78.491684),
    "B": (17.395044, 78.486671),
    "C": (17.400000, 78.480000),
    "D": (17.410000, 78.500000),
    "E": (17.420000, 78.470000),
    "F": (17.430000, 78.505000),
    "G": (17.440000, 78.515000),
    "H": (17.450000, 78.525000)
}

def plot_city_leaflet(G, quantum_path=None, classical_path=None):
    m = folium.Map(location=[17.40, 78.49], zoom_start=12, control_scale=True)

    # Draw all edges
    for u, v in G.edges():
        folium.PolyLine(
            [node_coords[u], node_coords[v]],
            color="gray", weight=2, opacity=0.5
        ).add_to(m)

    # Draw classical path (orange)
    if classical_path and len(classical_path) > 1:
        folium.PolyLine(
            [node_coords[n] for n in classical_path],
            color="orange", weight=8, opacity=0.8, tooltip="Classical Path"
        ).add_to(m)

    # Draw quantum path (purple)
    if quantum_path and len(quantum_path) > 1:
        folium.PolyLine(
            [node_coords[n] for n in quantum_path],
            color="purple", weight=8, opacity=0.8, tooltip="Quantum Path"
        ).add_to(m)

    # Add node markers
    for n, (lat, lon) in node_coords.items():
        folium.CircleMarker(
            [lat, lon], radius=10, fill=True,
            color="blue", fill_opacity=0.7, popup=f"{n}: {node_labels[n]}"
        ).add_to(m)

    return m

# --------- Run Optimization ---------
if st.button("🧠 Find Optimal Path"):
    with st.spinner("Training ML model and optimizing path..."):
        ml_model = QuantumDeliveryMLModel()
        ml_model.train_models()
        quantum_finder = QuantumInspiredPathFinder(ml_model)
        quantum_finder.create_city_graph(nodes, edges)

        constraints = {
            "weather": weather,
            "time_period": time_of_day,
            "delivery_type": delivery_type,
            "vehicle_capacity": vehicle_capacity,
            "order_weight": order_weight
        }
        results = quantum_finder.compare_algorithms(start, end, constraints)

    # --------- Show Path Results ---------
    st.markdown('<div class="section-title">🗺 City Navigation Map</div>', unsafe_allow_html=True)
    st_folium(
        plot_city_leaflet(
            G,
            quantum_path=results['quantum']['path'],
            classical_path=results['classical']['path']
        ),
        width=800, height=600
    )

    # --------- Path Details ---------
    st.markdown('<div class="section-title">📊 Path Comparison Results</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 🌟 Quantum Algorithm")
        st.write(f"*Path:* {' → '.join(results['quantum']['path']) if results['quantum']['path'] else '-'}")
        st.write(f"*Distance:* {results['quantum']['distance']:.2f} km")
        st.write(f"*Time:* {results['quantum']['estimated_delivery_time']:.1f} min")
        st.write(f"*Fuel:* {results['quantum']['estimated_fuel_consumption']:.2f} L")
        st.write(f"*CO2:* {results['quantum']['estimated_co2_emission']:.2f} kg")
    with cols[1]:
        st.markdown("### 🔍 Traditional GPS")
        st.write(f"*Path:* {' → '.join(results['classical']['path']) if results['classical']['path'] else '-'}")
        st.write(f"*Distance:* {results['classical']['distance']:.2f} km")
        st.write(f"*Time:* {results['classical']['estimated_delivery_time']:.1f} min")
        st.write(f"*Fuel:* {results['classical']['estimated_fuel_consumption']:.2f} L")
        st.write(f"*CO2:* {results['classical']['estimated_co2_emission']:.2f} kg")

    if "improvements" in results:
        st.markdown('<div class="section-title">🎯 Quantum Optimization Benefits</div>', unsafe_allow_html=True)
        savings = results["improvements"]
        savings_cols = st.columns(3)
        savings_cols[0].metric("Time Saved", f"{savings['time_saved_minutes']:.1f} min", f"{savings['time_saved_percentage']:.1f}%")
        savings_cols[1].metric("Fuel Saved", f"{savings['fuel_saved_liters']:.2f} L", f"{savings['fuel_saved_percentage']:.1f}%")
        savings_cols[2].metric("CO2 Reduced", f"{savings['co2_saved_kg']:.2f} kg", f"{savings['co2_saved_percentage']:.1f}%")

st.info("Configure your delivery scenario above and click '🧠 Find Optimal Path' to view results and map.")

st.markdown("---")
st.caption("Built with ❤ using Streamlit, Folium, scikit-learn, and NetworkX.")
