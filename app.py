import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
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

    def plot_city_graph(G, quantum_path=None, classical_path=None):
        # Use spring_layout for nice layout
        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Node positions
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node} ({node_labels[node]})")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(n) for n in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=False,
                color='#667eea',
                size=28,
                line_width=2
            ),
            hovertext=node_text,
            hoverinfo='text'
        )

        fig = go.Figure([edge_trace, node_trace])

        # Add quantum path (purple) and classical path (orange)
        if quantum_path and len(quantum_path) > 1:
            qp_x = [pos[n][0] for n in quantum_path]
            qp_y = [pos[n][1] for n in quantum_path]
            fig.add_trace(go.Scatter(
                x=qp_x, y=qp_y,
                mode='lines+markers',
                line=dict(width=6, color='#764ba2'),
                marker=dict(size=32, color='#764ba2', symbol="diamond"),
                name='Quantum Path'
            ))
        if classical_path and len(classical_path) > 1:
            cp_x = [pos[n][0] for n in classical_path]
            cp_y = [pos[n][1] for n in classical_path]
            fig.add_trace(go.Scatter(
                x=cp_x, y=cp_y,
                mode='lines+markers',
                line=dict(width=6, color='#f39c12'),
                marker=dict(size=32, color='#f39c12', symbol="circle"),
                name='Classical Path'
            ))

        fig.update_layout(
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    st.plotly_chart(
        plot_city_graph(
            G,
            quantum_path=results['quantum']['path'],
            classical_path=results['classical']['path']
        ),
        use_container_width=True
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

else:
    st.info("Configure your delivery scenario above and click '🧠 Find Optimal Path' to view results and map.")

st.markdown("---")
st.caption("Built with ❤ using Streamlit, Plotly, scikit-learn, and NetworkX.")
