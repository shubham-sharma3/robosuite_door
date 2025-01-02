# sac_test.py

import os
import torch
from sac_torch import SACAgent
import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
import time

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # =======================
    # Configuration Parameters
    # =======================

    # Random Seed for Reproducibility
    SEED = 42

    # Environment Configuration
    ENV_NAME = "Door"
    ROBOTS = ["Panda"]
    CONTROLLER = "JOINT_VELOCITY"
    HAS_RENDERER = True            # Enable rendering for visualization
    USE_CAMERA_OBS = False
    HORIZON = 300
    REWARD_SHAPING = True
    CONTROL_FREQ = 20

    # Testing Configuration
    TEST_EPISODES = 5               # Number of test episodes to run

    # SAC Hyperparameters (should match training)
    ALPHA = 0.2
    GAMMA = 0.99
    TAU = 0.005
    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4
    MAX_SIZE = 1000000
    BATCH_SIZE = 256

    # Model Checkpoint Path
    MODEL_PATH = "tmp/sac/actor.pth"

    # =======================
    # Initialize Environment
    # =======================

    # Set random seeds
    set_seed(SEED)

    # Create Robosuite environment with rendering enabled
    env = suite.make(
        ENV_NAME,
        robots=ROBOTS,
        controller_configs=suite.load_controller_config(default_controller=CONTROLLER),
        has_renderer=HAS_RENDERER,
        use_camera_obs=USE_CAMERA_OBS,
        horizon=HORIZON,
        reward_shaping=REWARD_SHAPING,
        control_freq=CONTROL_FREQ,
    )

    # Wrap the environment with GymWrapper for compatibility
    env = GymWrapper(env)

    # Extract environment dimensions
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    # =======================
    # Initialize SAC Agent
    # =======================

    # Define device (GPU if available, else CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize SAC agent with the defined hyperparameters
    agent = SACAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        max_size=MAX_SIZE,
        batch_size=BATCH_SIZE,
        device=device
    )

    # Load the trained Actor model
    if os.path.exists(MODEL_PATH):
        agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        agent.actor.eval()  # Set the Actor network to evaluation mode
        print("Actor model loaded successfully.")
    else:
        print(f"Actor model not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
        return

    # =======================
    # Testing Loop
    # =======================

    for episode in range(1, TEST_EPISODES + 1):
        state, _ = env.reset()
        done = False
        score = 0

        print(f"\nStarting Test Episode: {episode}")

        while not done:
            # Choose action without exploration noise (deterministic)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mean, std = agent.actor(state_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]  # Deterministic action (mean action)

            # Execute action in the environment
            next_state, reward, done, extra, info = env.step(action)
            score += reward

            # Update state
            state = next_state

            # Optional: Control the rendering speed
            time.sleep(0.02)  # Sleep for 20ms to control rendering speed (~50 FPS)

        print(f"Test Episode: {episode} | Score: {score}")

    # Close the environment after testing
    env.close()
    print("\nTesting completed.")

if __name__ == '__main__':
    main()