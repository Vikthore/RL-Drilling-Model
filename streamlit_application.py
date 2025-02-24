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

        ## Upload Your Dataset for Prediction
        You can upload your own dataset with **WOB, RPM, and MW** values to make predictions using the selected RL model.
  

        ## Understanding the Results
        Below, you will find the best drilling parameters as predicted by the trained RL model. These values 
        represent the optimal settings for drilling operations, helping to achieve higher efficiency and minimal bit wear.

        The displayed results include:
        - **Weight on Bit (WOB)**: The optimal force applied on the drill bit.
        - **Rotations Per Minute (RPM)**: The best rotational speed for drilling efficiency.
        - **Mud Weight**: The best drilling fluid weight to enhance performance.
        
        The table and bar chart provide an intuitive way to interpret these optimal values.

    """)
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV Format)", type=["csv"])
    user_data = None
    
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        expected_columns = ["WOB", "RPM", "MW"]
        if all(col in user_data.columns for col in expected_columns):
            st.success("✅ File uploaded successfully! Using custom dataset.")
        else:
            st.error(f"Uploaded dataset is missing required columns: {expected_columns}")
            user_data = None
    
    env = DrillingEnv() if user_data is None else DrillingEnv(data_path=uploaded_file)
    obs, _ = env.reset()
    
    
    try:
        if selected_agent == "PPO":
            model = PPO.load("ppo_drilling_agent")
        elif selected_agent == "A2C":
            model = A2C.load("a2c_drilling_agent")
        elif selected_agent == "DDPG":
            model = DDPG.load("ddpg_drilling_agent")
        else:
            raise ValueError("Invalid agent selection")
        
        action, _ = model.predict(obs, deterministic=True)
        # Ensure action is a NumPy array and reshape for compatibility
        # Ensure action is properly shaped
        # Ensure action is in the expected range before inverse transformation
        action = np.array(action).reshape(1, -1)
        
        # Rescale action from [-0.1, 0.1] to [0,1] for MinMaxScaler
        scaled_action = (action + 0.1) / 0.2  
        
        # Create a 5-feature placeholder
        full_scaled_action = np.zeros((1, 5))  
        full_scaled_action[0, :3] = scaled_action  # Assign only WOB, RPM, MW
        
        # Apply inverse transformation to get original values
        inverse_transformed = env.feature_scaler.inverse_transform(full_scaled_action)[0]
        
        # Extract WOB, RPM, MW
        best_wob = inverse_transformed[0]
        best_rpm = inverse_transformed[1]
        best_mw = inverse_transformed[2]

        

 # Extract only 3 parameters
    except Exception as e:
        st.error(f"Error loading {selected_agent} model: {e}")
        best_wob, best_rpm, best_mw = 0, 0, 0  # Default values in case of failure
    
    parameters_df = pd.DataFrame({
        "Parameter": ["WOB", "RPM", "Mud Weight"],
        "Optimal Value": [best_wob, best_rpm, best_mw]
    })
    
    st.subheader(f"Optimal Drilling Parameters - {selected_agent}")
    st.table(parameters_df)
    
    fig = px.bar(parameters_df, x="Parameter", y="Optimal Value", title=f"Optimal Drilling Parameters - {selected_agent}")
    st.plotly_chart(fig, use_container_width=True)

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
                model = PPO.load("ppo_drilling_agent") # Corrected model name
            elif selected_agent == "A2C":
                model = A2C.load("a2c_drilling_agent")
            elif selected_agent == "DDPG":
                model = DDPG.load("ddpg_drilling_agent")
            else:
                st.error("Invalid agent selected.")
                st.stop()

            env = DrillingEnv() # Use default DrillingEnv for evaluation
            episodes_to_evaluate = 10
            mean_reward = 0

            for episode in range(episodes_to_evaluate):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action) # Corrected unpacking
                    done = terminated or truncated
                    episode_reward += reward
                mean_reward += episode_reward
                st.write(f"Episode {episode+1} Reward: {episode_reward}")

            mean_reward /= episodes_to_evaluate
            st.success(f"🎯 Mean Reward ({selected_agent}) over {episodes_to_evaluate} episodes: {mean_reward}")
            env.close()

#Interactive Training
elif page == "🛠️ Interactive Training":
    st.image("possessed-photography-YKW0JjP7rlU-unsplash.jpg", use_container_width=True)
    st.header("🛠️ Train Your RL Model")
    st.subheader("Customize Training Environment")

    training_data_upload = st.file_uploader("Upload Training Data (CSV)", type=["csv"], key="training_uploader")
    train_on_custom_data = st.checkbox("Train agent on uploaded data?")
    custom_training_df = None

    if train_on_custom_data:
        if training_data_upload is not None:
            custom_training_df = pd.read_csv(training_data_upload)
            expected_columns_train = ["WOB", "RPM", "MW", "FlowRate", "Torque", "ROP", "Bit_Wear"] # Adjust to your CSV columns
            if all(col in custom_training_df.columns for col in expected_columns_train):
                st.success("✅ Training data loaded. Agent will be trained on your data.")
                st.dataframe(custom_training_df.head()) # Display uploaded data
            else:
                st.error(f"Training dataset must contain columns: {expected_columns_train}. Using default training environment.")
                custom_training_df = None # Fallback to default env
        else:
            st.warning("Please upload a CSV file for custom training data, or uncheck 'Train agent on uploaded data' to use default environment.")
            custom_training_df = None # Ensure no custom data is used if no file uploaded

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
            env = DrillingEnv(data_path=training_data_upload) if train_on_custom_data and custom_training_df is not None else DrillingEnv() # Use custom data if available and checkbox is checked
            if selected_agent == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            elif selected_agent == "A2C":
                model = A2C("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            elif selected_agent == "DDPG":
                model = DDPG("MlpPolicy", env, verbose=1, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma)
            model.learn(total_timesteps=total_timesteps)
            model.save(f"{selected_agent.lower()}_drilling_agent")
            st.session_state["trained_model"] = model # Store trained model in session state
        st.success(f"🎉 Training completed! {selected_agent} model saved.")
        st.download_button(
            label="⬇️ Download Trained Model",
            data=open(f"{selected_agent.lower()}_drilling_agent.zip", "rb").read(),
            file_name=f"{selected_agent.lower()}_drilling_agent.zip",
            mime="application/zip"
        )

# --- Data Analysis ---
if page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    st.image("data_analytics.jpeg", use_container_width=True)
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

# --- Compare Agents ---
elif page == "🤖 Compare Agents":
    st.header("🤖 Compare RL Agents")
    st.image("group_of_robots.jpg", use_container_width=True)
    agent_types = ["PPO", "A2C", "DDPG"]
    mean_rewards = {}
    episodes_compare = st.slider("Number of Episodes for Comparison", 10, 50, 20) # Slider for episodes

    if st.button("🚀 Run Comparison"):
        with st.spinner("Comparing agents..."):
            for agent_type in agent_types:
                if agent_type == "PPO":
                    model = PPO.load("ppo_drilling_agent")
                elif agent_type == "A2C":
                    model = A2C.load("a2c_drilling_agent")
                elif agent_type == "DDPG":
                    model = DDPG.load("ddpg_drilling_agent")

                env = DrillingEnv()
                episodes_to_evaluate = episodes_compare # Use slider value
                total_reward = 0
                for episode in range(episodes_to_evaluate):
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        total_reward += reward
                mean_reward = total_reward / episodes_to_evaluate
                mean_rewards[agent_type] = mean_reward
                env.close()

        comparison_df = pd.DataFrame(list(mean_rewards.items()), columns=['Agent', 'Mean Reward'])
        fig = px.bar(comparison_df, x="Agent", y="Mean Reward",
                     title="Mean Reward Comparison Across Agents")
        st.plotly_chart(fig, use_container_width=True)
        st.write("### Agent Comparison Results")
        st.dataframe(comparison_df)
        
# --- Learning Resources ---
elif page == "📚 Resources":
    st.header("📖 Learning Resources")
    st.image("altumcode-oZ61KFUQsus-unsplash.jpg", use_container_width=True)
    st.markdown("- 🔗 [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📚 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
    st.markdown("- 🏗️ [GitHub Repository](https://github.com/your-repo)")

st.write("🌎 Website last updated on ", datetime.date.today())
