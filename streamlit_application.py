import streamlit as st
import os
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
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e3a5f;
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
    
    # Agent Selection
    st.header("🤖 Agent Selection")
    selected_agent = st.selectbox("Choose Agent", ["PPO", "A2C", "DDPG"])

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
        with st.spinner(f"Evaluating {selected_agent} agent..."):
            model = None
            if selected_agent == "PPO":
                model = PPO.load("ppo_drilling_agents")
            elif selected_agent == "A2C":
                model = A2C.load("a2c_drilling_agent")
            elif selected_agent == "DDPG":
                model = DDPG.load("ddpg_drilling_agent")
            else:
                st.error("Invalid agent selected.")
                st.stop()
            
            env = DrillingEnv()
            episodes_to_evaluate = 10
            mean_reward = 0
            
            for episode in range(episodes_to_evaluate):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                mean_reward += episode_reward
                st.write(f"Episode {episode+1} Reward: {episode_reward}")
            
            mean_reward /= episodes_to_evaluate
            st.success(f"🎯 Mean Reward ({selected_agent}) over {episodes_to_evaluate} episodes: {mean_reward}")
            env.close()

# --- Data Analysis Section ---
elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    st.subheader("Explore and Analyze Drilling Data")
    
    df = pd.read_csv("synthetic_drilling_data.csv")
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Descriptive Statistics")
    st.write(df.describe())
    
    st.write("### Interactive Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.columns)
    
    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Download Analysis Results")
    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False),
        file_name="drilling_data_analysis.csv",
        mime="text/csv"
    )

# --- Learning Resources ---
elif page == "📚 Resources":
    st.header("📖 Learning Resources")
    st.markdown("- 🔗 [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📚 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
    st.markdown("- 🏗️ [GitHub Repository](https://github.com/your-repo)")

st.write("🌎 Website last updated on ", datetime.date.today())
