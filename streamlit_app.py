import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np  # Make sure numpy is imported if you use it in your env

# --- Page Configuration ---
st.set_page_config(
    page_title="RL Drilling Agent Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Website Title and Introduction ---
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
    """) # Replace with your detailed project introduction

# --- Agent Performance Metrics ---
st.header("Agent Performance Evaluation")
st.subheader("Key Metric: Episode Reward Mean")

mean_reward = 146349.72  # **REPLACE WITH YOUR ACTUAL FINAL EP_REW_MEAN VALUE**
st.metric(label="Average Episode Reward (ep_rew_mean)", value=f"{mean_reward:.2f}")

# --- Training Curve Chart (Conditional Display) ---
st.header("Training Curve")

# Example data - **REPLACE WITH YOUR ACTUAL TRAINING DATA IF YOU HAVE IT SAVED!**
training_iterations = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
ep_rew_means = [125000, 130000, 138000, 142000, 144500, 145800, 146200, 146300, 146340, 146349]

if training_iterations and ep_rew_means:
    st.subheader("Episode Reward Mean over Training")
    fig, ax = plt.subplots()
    ax.plot(training_iterations, ep_rew_means)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Episode Reward Mean")
    st.pyplot(fig)
else:
    st.write("*(Training curve data not available for display. To show a chart, replace the example data above with your actual training history.)*")

# --- Code Examples ---
st.header("Code Examples")

st.subheader("Drilling Environment (`DrillingEnv` class) - Partial Example")
code_env = """
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DrillingEnv(gym.Env):
    def __init__(self):
        super(DrillingEnv, self).__init__()
        # Define action and observation space (replace with your actual spaces)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.max_steps_per_episode = 5000
        self.current_step = 0
        # ... (Initialize your environment state, scalers, etc.) ...

    def step(self, action):
        # --- 1. Action Application ---
        current_state_scaled = self.scaled_features[self.current_step]
        action_unscaled = action # Assuming action is in scaled space
        next_state_scaled = current_state_scaled + action_unscaled[:len(self.action_space.shape)]
        next_state_scaled = np.clip(next_state_scaled, 0.0, 1.0)

        # --- 4. Unscale the Next State ---
        next_state_unscaled = self.feature_scaler.inverse_transform(next_state_scaled.reshape(1, -1)).flatten()
        next_wob, next_rpm, next_mw, next_flowrate, next_torque = next_state_unscaled

        # --- 5. Calculate Next ROP and BitWear ---
        k1, k2, k3, k4, k5, k6 = 5, 0.5, 0.3, 0.1, 0.01, 0.01 # Example constants - replace with your values
        next_rop = k1 * (next_wob ** k2) * (next_rpm ** k3) * np.exp(-k4 * next_mw) + np.random.normal(0, 2)
        next_bitwear = k5 * (next_wob + next_torque) - k6 * next_rop + np.random.normal(0, 0.05)

        next_rop_clipped = np.clip(next_rop, 0, 50)
        next_bitwear_clipped = np.clip(next_bitwear, 0, 1)

        # --- 6. Scale the Next State (for observation) ---
        next_observation = next_state_scaled

        # --- 3. Reward Calculation ---
        alpha = 1.0; beta = 1.0 # Example reward weights - replace with your values
        reward = alpha * next_rop_clipped - beta * next_bitwear_clipped

        # --- 8. Episode Termination ---
        terminated = False; truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps_per_episode -1:
            truncated = True
        info = {}
        return next_observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # ... (Your reset method code) ...
        return self.observation, self.info

    def render(self):
        # ... (Your render method code if you have one) ...
        pass

    def close(self):
        super().close()
"""
st.code(code_env, language='python')

st.subheader("PPO Agent Training - Example")
code_training = """
import gymnasium as gym
from stable_baselines3 import PPO
from drilling_env import DrillingEnv

env = DrillingEnv()
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001) # Example learning rate
model.learn(total_timesteps=1000000)
model.save("ppo_drilling_agent")
"""
st.code(code_training, language='python')

# --- Conclusion and Future Work ---
st.header("Conclusion and Future Work")
st.write("""
    This project demonstrates the successful application of Reinforcement Learning for
    automated drilling control in a simulated environment. The trained PPO agent shows promising
    performance in optimizing drilling parameters to achieve a balance between Rate of Penetration and BitWear.

    ```
    However, there are several avenues for future research and improvement:

    *   Reward Function Refinement: Experiment with different reward function designs,
        especially by adjusting the weights for ROP and BitWear to better reflect real-world drilling priorities.
    *   Hyperparameter Optimization: Conduct a more systematic hyperparameter search for the PPO agent
        to potentially achieve even higher performance and training stability.
    *   Advanced Environment Features: Enhance the `DrillingEnv` simulation by incorporating more realistic
        drilling dynamics, geological variations, and failure modes to create a more challenging and representative training ground.
    *   Transfer Learning and Real-World Data: Explore the possibility of using real-world drilling data
        to pre-train or fine-tune the RL agent, potentially improving its generalization and applicability to real-world scenarios.
    *   Visualization and Explainability: Develop more advanced visualizations to understand the agent's decision-making process
        and gain insights into optimal drilling strategies learned by the RL agent.
    ```
    """) # Customize with your actual conclusions and future directions

# --- Resources and Links ---
st.header("Resources")
st.markdown("[Streamlit Documentation](https://docs.streamlit.io/)")
st.markdown("[Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)")
# st.markdown("[Link to your GitHub Repository](Your GitHub Repo URL)") # Add your GitHub repo URL here if you have one

# --- Date (Optional) ---
st.write("Website created on ", st.date_input("Date", value=datetime.date.today()))