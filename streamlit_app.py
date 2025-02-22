import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO, A2C, DDPG
from drilling_env import DrillingEnv

# --- Page Configuration ---
st.set_page_config(
    page_title="RL Drilling Agent Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
    }
    .main-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton button {
        background-color: #e74c3c;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #c0392b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation with Icons ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/61/Oil_rig.jpg", width=250)
    st.title("🌍 Navigation")
    page = st.radio("Go to", [
        "🏠 Project Overview", 
        "📈 Agent Performance", 
        "🛠️ Interactive Training", 
        "📊 Data Analysis", 
        "🤖 Compare Agents", 
        "📚 Resources"
    ])
    
    # User Authentication
    st.header("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state["authenticated"] = True
            st.success("✅ Logged in successfully!")
        else:
            st.error("❌ Invalid username or password")

# --- Main Content Area ---
if page == "🏠 Project Overview":
    st.title("Reinforcement Learning for Automated Drilling Control")
    st.image("https://miro.medium.com/max/1400/1*5VQ14OCOm2JdFYPoRZdF7g.png", use_container_width=True)
    st.write("This dashboard provides an interactive experience for monitoring and training RL models for drilling optimization.")
    
elif page == "📈 Agent Performance":
    st.header("Agent Performance Evaluation")
    mean_reward = 146349.72
    st.metric(label="🏆 Average Episode Reward", value=f"{mean_reward:.2f}")
    
    st.subheader("Training Progress")
    training_iterations = np.linspace(10000, 100000, 10).astype(int)
    ep_rew_means = np.linspace(125000, 146349, 10).astype(int)
    fig = px.line(x=training_iterations, y=ep_rew_means, labels={"x": "Training Timesteps", "y": "Episode Reward Mean"}, title="Training Progress")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🛠️ Interactive Training":
    if not st.session_state.get("authenticated", False):
        st.warning("🔐 Please log in to access this feature.")
    else:
        st.header("🎯 Train Your RL Model")
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001)
            ent_coef = st.slider("Entropy Coefficient", 0.0, 0.2, 0.01)
        with col2:
            total_timesteps = st.slider("Total Timesteps", 10000, 1000000, 100000)
            gamma = st.slider("Discount Factor (Gamma)", 0.9, 0.99, 0.99)
        
        if st.button("🚀 Start Training"):
            with st.spinner("⏳ Training in progress..."):
                env = DrillingEnv()
                model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
                model.learn(total_timesteps=total_timesteps)
                model.save("ppo_drilling_agent")
            st.success("🎉 Training completed! Model saved.")

elif page == "📊 Data Analysis":
    st.header("📌 Data Insights")
    uploaded_file = st.file_uploader("Upload your drilling data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 📂 Data Preview")
        st.dataframe(df.head())
        st.write("### 📊 Statistical Summary")
        st.write(df.describe())
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="ROP", nbins=20, labels={"ROP": "Rate of Penetration"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(df, y="BitWear", labels={"BitWear": "Bit Wear"})
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Compare Agents":
    st.header("⚖️ Compare RL Models")
    algorithms = ["PPO", "A2C", "DDPG"]
    selected_algorithms = st.multiselect("Select Algorithms to Compare", algorithms, default=["PPO"])
    if st.button("🔍 Run Comparison"):
        with st.spinner("⏳ Running comparison..."):
            env = DrillingEnv()
            results = {}
            for algo in selected_algorithms:
                model = PPO('MlpPolicy', env, verbose=1) if algo == "PPO" else A2C('MlpPolicy', env, verbose=1) if algo == "A2C" else DDPG('MlpPolicy', env, verbose=1)
                model.learn(total_timesteps=100000)
                results[algo] = model
            st.write("### 📊 Comparison Results")
            for algo, model in results.items():
                st.write(f"#### {algo} Performance")
                st.write(f"🏆 Average Reward: {model.episode_reward_mean}")
