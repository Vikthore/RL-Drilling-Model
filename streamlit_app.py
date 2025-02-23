import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO, A2C, DDPG
from drilling_env.py import DrillingEnv

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
        background-color: #1e3a5f;
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
    st.image("oil_rig.jpeg", width=250)
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
    st.title("🚀 Reinforcement Learning for Smart Drilling")
    st.image("oil_rig.jpeg", use_container_width=True)
    st.write("""
        Welcome to the **AI-powered Drilling Optimization Dashboard**! 🌍🛢️
        
        This platform provides an **interactive experience** for monitoring and training RL models 
        to **optimize drilling performance** by maximizing **Rate of Penetration (ROP)** while minimizing **Bit Wear**.
        
        🔥 **Key Features:**
        - 📊 **Monitor RL Agent Performance** with real-time updates
        - 🎯 **Interactive Training Module** to fine-tune hyperparameters
        - 📈 **Live Data Analysis** for drilling insights
        - 🤖 **Compare Multiple RL Algorithms** for efficiency
    """)
    
    
# --- Agent Performance Evaluation ---
if page == "📈 Agent Performance":
    st.header("Agent Performance Evaluation")
    st.subheader("Evaluate Trained RL Agent")
    
    if st.button("🚀 Run Evaluation"):
        with st.spinner("Evaluating the agent..."):
            model = PPO.load("ppo_drilling_agent")
            env = DrillingEnv()
            episodes_to_evaluate = 10
            mean_reward = 0
            
            for episode in range(episodes_to_evaluate):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                mean_reward += episode_reward
                st.write(f"Episode {episode+1} Reward: {episode_reward}")
            
            mean_reward /= episodes_to_evaluate
            st.success(f"🎯 Mean Reward over {episodes_to_evaluate} episodes: {mean_reward}")
            env.close()

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

elif page == "📚 Resources":
    st.header("📖 Learning Resources")
    st.markdown("- 🔗 [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📚 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
    st.markdown("- 🏗️ [GitHub Repository](https://github.com/your-repo)")

st.write("🌎 Website last updated on ", st.date_input("Date", value=datetime.date.today()))
