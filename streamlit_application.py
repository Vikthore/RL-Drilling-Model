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
    st.image("tricone_bit.jpeg", width=250)
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
        
       ## What is Reinforcement Learning (RL)?
        Reinforcement Learning (RL) is a type of machine learning where an agent learns to make optimal decisions 
        by interacting with an environment and receiving feedback in the form of rewards or penalties. 
        RL is widely used in industries such as robotics, finance, healthcare, and **oil & gas drilling optimization**.
        
        ## How RL is Used in This Application
        This platform enables users to train RL agents to optimize **Rate of Penetration (ROP)** while minimizing **Bit Wear**.
        The trained agent can help predict the best **Weight on Bit (WOB), Rotations Per Minute (RPM), and Mud Weight (MW)** 
        for efficient drilling operations.
        
        🔥 **Key Features:**
        - 📊 **Monitor RL Agent Performance** with real-time updates
        - 🎯 **Interactive Training Module** to fine-tune hyperparameters
        - 📈 **Live Data Analysis** for drilling insights
        - 🤖 **Train on Custom Data** for personalized optimization
        
        ## How to Interpret Results
        - **Episode Reward**: A higher reward indicates better drilling efficiency with lower bit wear.
        - **Training Progress**: Shows how the RL model improves over iterations.
        - **Agent Comparison**: Evaluate different RL algorithms (PPO, A2C, DDPG) to determine the most effective one.
        
        ## RL Interpretability in the Industry
        - **Operational Decision-Making**: RL can assist drilling engineers in optimizing parameters dynamically.
        - **Cost Reduction**: By minimizing bit wear, companies can reduce downtime and increase efficiency.
        - **Safety Improvements**: Optimized drilling processes reduce the risk of failures and environmental hazards.
 
    """)
        if "trained_model" in st.session_state:
            env = DrillingEnv()
            obs, _ = env.reset()
            action, _ = st.session_state["trained_model"].predict(obs, deterministic=True)
            best_wob, best_rpm, best_torque, best_flowrate, best_mudweight = action
            
            parameters_df = pd.DataFrame({
                "Parameter": ["WOB", "RPM", "Torque", "Flow Rate", "Mud Weight"],
                "Optimal Value": [best_wob, best_rpm, best_torque, best_flowrate, best_mudweight]
            })
            
            st.table(parameters_df)
            
            fig = px.bar(parameters_df, x="Parameter", y="Optimal Value", title="Optimal Drilling Parameters")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trained model available. Please train an agent in the Interactive Training section.")


if page == "📈 Agent Performance":
    st.header("Agent Evaluation & Training on Custom Data")
    st.image("julien-tromeur-FYOwBvRb2Mk-unsplash.jpg", use_container_width=True)
    st.subheader("Upload Your Dataset to Train the RL Model")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    custom_df = None
    if uploaded_file is not None:
        custom_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview")
        st.dataframe(custom_df.head())
    
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
            
            env = DrillingEnv(custom_df=custom_df) if custom_df is not None else DrillingEnv()
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

# --- Interactive Training ---
if page == "🛠️ Interactive Training":
    st.header("🛠️ Train Your RL Model")
    st.subheader("Set Training Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("Learning Rate", 0.00001, 0.001, 0.0001)
        gamma = st.slider("Discount Factor (Gamma)", 0.9, 0.99, 0.99)
    with col2:
        total_timesteps = st.slider("Total Timesteps", 10000, 1000000, 100000)
        ent_coef = st.slider("Entropy Coefficient", 0.0, 0.2, 0.01)
    
    if st.button("🚀 Start Training"):
        with st.spinner("Training in progress..."):
            env = DrillingEnv()
            if selected_agent == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            elif selected_agent == "A2C":
                model = A2C("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            elif selected_agent == "DDPG":
                model = DDPG("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            model.learn(total_timesteps=total_timesteps)
            model.save(f"{selected_agent.lower()}_drilling_agent")
        st.success(f"🎉 Training completed! {selected_agent} model saved.")


# --- Data Analysis ---
if page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    st.subheader("Upload Your Dataset for Analysis")
    
    uploaded_data = st.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_data is not None:
        analysis_df = pd.read_csv(uploaded_data)
        st.write("### Uploaded Dataset Preview")
        st.dataframe(analysis_df.head())
    else:
        analysis_df = pd.read_csv("synthetic_drilling_data.csv")
        st.write("### Default Dataset: Synthetic Drilling Data")
        st.dataframe(analysis_df.head())
    
    if analysis_df is not None:
        st.subheader("Dataset Statistics")
        st.write(analysis_df.describe())
        
        st.subheader("Correlation Heatmap")
        numeric_df = analysis_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Interactive Scatter Plot")
        x_axis = st.selectbox("Select X-axis", analysis_df.columns)
        y_axis = st.selectbox("Select Y-axis", analysis_df.columns)
        scatter_fig = px.scatter(analysis_df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(scatter_fig, use_container_width=True)

# --- Learning Resources ---
elif page == "📚 Resources":
    st.header("📖 Learning Resources")
    st.markdown("- 🔗 [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📚 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
    st.markdown("- 🏗️ [GitHub Repository](https://github.com/your-repo)")

st.write("🌎 Website last updated on ", datetime.date.today())
