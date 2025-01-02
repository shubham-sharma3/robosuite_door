import os
import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. Training will use GPU.")
    else:
        print(f"CUDA is NOT available. Training will use CPU.")

    # Create directory for SAC models if it doesn't exist
    if not os.path.exists("tmp/sac"):
        os.makedirs("tmp/sac")

    # Define environment
    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
    env = DummyVecEnv([lambda: env])  # Wrap the environment to use vectorized version (Stable-Baselines3 requirement)

    # SAC Hyperparameters
    actor_lr = 3e-4
    critic_lr = 3e-4
    batch_size = 256
    buffer_size = 1000000
    gamma = 0.99
    tau = 0.005
    ent_coef = "auto"  # Automatically adjusted by SAC
    total_timesteps = 100000  # Number of training timesteps

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",  # Policy type (Multi-layer perceptron)
        env,
        learning_rate=actor_lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log="./tmp/sac/",  # Log results to tensorboard
    )

    # Evaluate the model periodically
    eval_callback = EvalCallback(
        env,
        best_model_save_path="tmp/sac/",
        log_path="tmp/sac/eval_logs/",
        eval_freq=5000,  # Evaluate every 5000 timesteps
        deterministic=True,
        render=False,  # Disable rendering during evaluation
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the model after training
    model.save("sac_door_model")
    print("Training complete. Model saved.")

    # Optionally, visualize the trained agent
    model = SAC.load("sac_door_model")
    obs = env.reset()
    for _ in range(1000):  # Run for 1000 steps or more to visualize the behavior
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()  # This will render the robot interacting with the environment
        if dones:
            break
    env.close()



