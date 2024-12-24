import robosuite as suite
from robosuite.controllers import load_controller_config
from Bin_picking import BinPickingEnv


def main():
    controller_config = load_controller_config(default_controller="IK_POSE")

    env = BinPickingEnv(
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        use_camera_obs=False,
        reward_scale=1.0
    )

    obs = env.reset()

    for _ in range(200):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print("Reward:", reward)

    env.close()


if __name__ == "__main__":
    main()
