import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == '__main__':

    if not os.path.exists("tmp_2048batch_size/td3"):
        os.makedirs("tdm_2048batch_size/td3")
    else:
        pass

    env_name = "Lift"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )


    env = GymWrapper(env)

    actor_learning_rate = 0.0003
    critic_learning_rate = 0.001
    batch_size = 2048
    layer1_size = 512
    layer2_size = 512

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size,
                  batch_size=batch_size)

    n_games = 3
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_leanring_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation= env.reset()
        # print(f"Observation  after reset: {observation}")

        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            step_result = env.step(action)
            next_observation, reward, done, info = step_result if len(step_result) == 4 else step_result[:4]
            env.render()
            score += reward

            observation = next_observation

        print(f"Episode: {i} Score: {score}")