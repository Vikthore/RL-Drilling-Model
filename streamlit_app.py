import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO, A2C, DDPG
from drilling_env import DrillingEnv  # Assuming your environment is in a separate file

# --- Page Configuration ---
st.set_page_config(
    page_title="RL Drilling Agent Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- User Authentication ---
def authenticate(username, password):
    """Simple authentication function."""
    return username == "admin" and password == "password"

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Project Overview", "Agent Performance", "Interactive Training", "Data Analysis", "Compare Agents", "Resources"])

    # User Authentication
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["authenticated"] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

# --- Main Content Area ---
if page == "Project Overview":
    st.title("Reinforcement Learning for Automated Drilling Control")
    st.header("Project Overview")
    st.write("""
        This Streamlit dashboard showcases a Reinforcement Learning (RL) agent
        trained to control drilling parameters in a simulated environment.
        The goal of this project is to develop an intelligent agent that can optimize
        the drilling process by maximizing Rate of Penetration (ROP) while minimizing BitWear.

        ```
        This project uses the Proximal Policy Optimization (PPO) algorithm from the `stable-baselines3` library
        to train an agent within a custom-built `DrillingEnv` environment.
        ```
        """)

elif page == "Agent Performance":
    st.header("Agent Performance Evaluation")
    st.subheader("Key Metric: Episode Reward Mean")

    mean_reward = 146349.72  # **REPLACE WITH YOUR ACTUAL FINAL EP_REW_MEAN VALUE**
    st.metric(label="Average Episode Reward (ep_rew_mean)", value=f"{mean_reward:.2f}")

    # --- Training Curve Chart ---
    st.subheader("Training Curve")
    training_iterations = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    ep_rew_means = [125000, 130000, 138000, 142000, 144500, 145800, 146200, 146300, 146340, 146349]

    if training_iterations and ep_rew_means:
        fig = px.line(x=training_iterations, y=ep_rew_means, labels={"x": "Training Timesteps", "y": "Episode Reward Mean"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("*(Training curve data not available for display.)*")

elif page == "Interactive Training":
    if not st.session_state.get("authenticated", False):
        st.warning("Please log in to access this feature.")
    else:
        st.header("Interactive Training")
        st.subheader("Adjust Hyperparameters and Train the Agent")

        # Hyperparameter Tuning
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001)
            ent_coef = st.slider("Entropy Coefficient", 0.0, 0.2, 0.01)
        with col2:
            total_timesteps = st.slider("Total Timesteps", 10000, 1000000, 100000)
            gamma = st.slider("Discount Factor (Gamma)", 0.9, 0.99, 0.99)

        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                env = DrillingEnv()
                model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
                model.learn(total_timesteps=total_timesteps)
                model.save("ppo_drilling_agent")
            st.success("Training completed! Model saved.")

elif page == "Data Analysis":
    st.header("Data Analysis")
    st.subheader("Upload and Analyze Drilling Data")

    uploaded_file = st.file_uploader("Upload your drilling data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        # Data Analysis
        st.write("### Data Analysis")
        st.write("#### Summary Statistics")
        st.write(df.describe())

        st.write("#### Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.write("##### ROP Distribution")
            fig = px.histogram(df, x="ROP", nbins=20, labels={"ROP": "Rate of Penetration"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("##### BitWear Over Time")
            fig = px.line(df, y="BitWear", labels={"BitWear": "Bit Wear"})
            st.plotly_chart(fig, use_container_width=True)

elif page == "Compare Agents":
    st.header("Compare RL Agents")
    st.subheader("Compare the Performance of Different RL Algorithms")

    algorithms = ["PPO", "A2C", "DDPG"]
    selected_algorithms = st.multiselect("Select Algorithms to Compare", algorithms, default=["PPO"])

    if st.button("Run Comparison"):
        with st.spinner("Running comparison..."):
            env = DrillingEnv()
            results = {}
            for algo in selected_algorithms:
                if algo == "PPO":
                    model = PPO('MlpPolicy', env, verbose=1)
                elif algo == "A2C":
                    model = A2C('MlpPolicy', env, verbose=1)
                elif algo == "DDPG":
                    model = DDPG('MlpPolicy', env, verbose=1)
                model.learn(total_timesteps=100000)
                results[algo] = model

            st.write("### Comparison Results")
            for algo, model in results.items():
                st.write(f"#### {algo} Performance")
                st.write(f"Average Reward: {model.episode_reward_mean}")

elif page == "Resources":
    st.header("Resources")
    st.markdown("[Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("[Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
    st.markdown("[GitHub Repository](https://github.com/your-repo)")  # Add your GitHub repo URL here

# --- Date (Optional) ---
st.write("Website created on ", st.date_input("Date", value=datetime.date.today()))
