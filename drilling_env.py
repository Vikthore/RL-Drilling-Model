import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Or StandardScaler

class DrillingEnv(gym.Env):
    """Custom Environment for RL-based Drilling Optimization."""

    metadata = {"render_modes": ["human"], "render_fps": 30} # For visualization (optional)

    def __init__(self, data_path='Desktop/synthetic_drilling_data.csv'):
        super().__init__()
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        # 1. Feature and Target Selection
        self.feature_columns = ['WOB', 'RPM', 'MW', 'FlowRate', 'Torque'] # Input features for state
        self.target_columns = ['ROP', 'BitWear'] # Target variables for reward

        self.features = self.df[self.feature_columns].values
        self.targets = self.df[self.target_columns].values

        # 2. Scaling - Initialize and Fit Scaler
        self.feature_scaler = MinMaxScaler() # Or StandardScaler()
        self.scaled_features = self.feature_scaler.fit_transform(self.features)

        # --- Define State Space and Action Space ---
        # 3.1. Observation Space (State Space)
        # Assuming state is composed of the scaled input features (WOB, RPM, MW, FlowRate, Torque)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.feature_columns),), dtype=np.float32) # Example bounds for MinMaxScaler

        # 3.2. Action Space
        # Define action space for adjusting WOB, RPM, and MW (example: adjustments in scaled space)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32) # Adjust WOB, RPM, MW within +/- 0.1 in scaled space

        # --- Environment State Initialization ---
        self.current_step = 0 # Track current step in episode
        self.max_steps_per_episode = len(self.df) # Example: Episode length = dataset size


    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed, options=options)
        self.current_step = 0 # Reset step counter
        initial_state = self.scaled_features[self.current_step] # Start from first row of scaled data
        observation = initial_state # Initial observation is the initial state
        info = {} # Additional info (optional)

        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""

        # --- 1. Action Application ---
        # Get current state (scaled features)
        current_state_scaled = self.scaled_features[self.current_step]

        print("Action:", action) # Debug print: Action value
        print("Current Scaled State:", current_state_scaled) # Debug print: Current scaled state

        # Unscale action (adjustments in scaled space) and apply to scaled state
        action_unscaled = action # For now, assuming action is already in scaled space.  Unscaling logic would go here if needed.
        next_state_scaled = current_state_scaled + action_unscaled[:len(self.action_space.shape)] # Apply action to WOB, RPM, MW (first 3 actions)

        # Clip scaled state to stay within [0, 1] bounds (if using MinMaxScaler) - adjust if using StandardScaler or different bounds
        next_state_scaled_before_clip = next_state_scaled.copy() # For debug printing before clipping
        next_state_scaled = np.clip(next_state_scaled, 0.0, 1.0)
        print("Next Scaled State (Before Clip):", next_state_scaled_before_clip) # Debug print: Scaled state before clip
        print("Next Scaled State (After Clip):", next_state_scaled) # Debug print: Scaled state after clip


        # --- 4. Unscale the Next State ---
        next_state_unscaled = self.feature_scaler.inverse_transform(next_state_scaled.reshape(1, -1)).flatten()
        next_wob, next_rpm, next_mw, next_flowrate, next_torque = next_state_unscaled # Unpack unscaled features
        print("Next Unscaled State (WOB, RPM, MW, FlowRate, Torque):", next_wob, next_rpm, next_mw, next_flowrate, next_torque) # Debug print: Unscaled state


        # --- 5. Calculate Next ROP and BitWear ---
        # ** Placeholder: Calculate next_rop using Bourgoyne & Young's model (or your ROP formula)
        k1, k2, k3, k4, k5, k6 = 5, 0.5, 0.3, 0.1, 0.01, 0.01  # Constants - ensure these are consistent with your data generation
        next_rop = k1 * (next_wob ** k2) * (next_rpm ** k3) * np.exp(-k4 * next_mw) + np.random.normal(0, 2) #  ROP formula from user

        # ** Placeholder: Calculate next_bitwear using your BitWear formula
        next_bitwear = k5 * (next_wob + next_torque) - k6 * next_rop + np.random.normal(0, 0.05) # BitWear formula from user

        print("Next ROP (Before Clip):", next_rop) # Debug print: ROP before clip
        print("Next BitWear (Before Clip):", next_bitwear) # Debug print: BitWear before clip

        # Ensure no negative values and clip BitWear
        next_rop_clipped = np.clip(next_rop, 0, 50) # Clip ROP to a reasonable range
        next_bitwear_clipped = np.clip(next_bitwear, 0, 1) # Clip BitWear to [0, 1]
        print("Next ROP (After Clip):", next_rop_clipped) # Debug print: ROP after clip
        print("Next BitWear (After Clip):", next_bitwear_clipped) # Debug print: BitWear after clip


        # --- 6. Scale the Next State (for observation) ---
        next_observation = next_state_scaled # For now, next_observation is just the scaled state. You might choose to include scaled ROP/BitWear in the state later.


        # --- 3. Reward Calculation ---
        alpha = 0.1 # Scaling factor for ROP (adjust as needed)
        beta = 1.0  # Scaling factor for BitWear (adjust as needed)
        reward = alpha * next_rop_clipped - beta * next_bitwear_clipped # Reward function
        print(f"Step Reward: {reward}")


        # --- 8. Episode Termination ---
        terminated = False # Set termination condition if needed (e.g., based on BitWear threshold)
        truncated = False # Episode truncation (e.g., max steps reached)
        self.current_step += 1 # Increment step counter
        if self.current_step >= self.max_steps_per_episode -1: # -1 because current_step starts at 0
            truncated = True


        info = {} # Additional information (optional)


        return next_observation, reward, terminated, truncated, info

    def render(self):
        """Optional: Render the environment (e.g., visualize drilling parameters)."""
        # Implement visualization if desired (e.g., using matplotlib)
        pass

    def close(self):
        """Optional: Clean up resources."""
        super().close()


# --- Example of Environment Usage ---
if __name__ == '__main__':
    env = DrillingEnv()
    observation, info = env.reset(seed=42)
    print("Initial observation:", observation)

    action = env.action_space.sample() # Take a random action
    next_observation, reward, terminated, truncated, info = env.step(action)
    print("\nAction:", action)
    print("Next observation:", next_observation)
    print("Reward:", reward)
    print("Terminated:", terminated, "Truncated:", truncated)

    env.close()
