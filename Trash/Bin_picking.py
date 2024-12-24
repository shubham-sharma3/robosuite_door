import numpy as np
import mujoco
from robosuite import load_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.robots import Panda
from robosuite.models import MujocoWorldBase
from robosuite.models.grippers import gripper_factory
from robosuite.utils.mjcf_utils import new_joint
from robosuite.utils.mjcf_utils import xml_path_completion


class BinPickingEnv(MujocoEnv):
    def __init__(self,
                 robots="Panda",
                 controller_configs=None,
                 gripper_type="PandaGripper",
                 has_renderer=False,
                 has_offscreen_renderer=False,
                 use_camera_obs=False,
                 reward_scale=1.0,
                 ):
        """
        BinPickingEnv: 一个机器人从箱子中抓取物体的环境。

        奖励函数设计思路：
        1. 距离奖励：鼓励机器人将夹爪移动至物体附近（距离越近，奖励越高）。
        2. 抓取奖励：如果机器人成功抓住一个球体（物体离开桌面并在夹爪附近），给予较大正奖励。
        3. 抬升奖励：抓住物体后，根据物体高度给予额外奖励。
        """

        if controller_configs is None:
            controller_configs = load_controller_config(default_controller="IK_POSE")

        # 定义机器人
        self.robot = Panda()
        # 添加夹爪
        gripper = gripper_factory(gripper_type)
        self.robot.add_gripper(gripper)

        # 创建世界
        world = MujocoWorldBase()
        self.robot.set_base_xpos([0, 0, 0])
        world.merge(self.robot)

        # 创建工作空间（桌子）
        arena = TableArena()
        arena.set_origin([0.8, 0, 0])
        world.merge(arena)

        # 记录桌面高度
        self.table_offset = 0.8  # 来自 arena.set_origin 的 z 坐标
        self.num_objects = 10
        self.objects = []

        # 添加物体（球）
        for i in range(self.num_objects):
            sphere = BallObject(
                name=f"sphere_{i}",
                size=[0.04],
                rgba=[0, 0.5, 0.5, 1]
            ).get_obj()
            pos = np.random.uniform(low=[0.75, -0.2, 0.8], high=[0.85, 0.2, 0.8])
            sphere.set('pos', f"{pos[0]} {pos[1]} {pos[2]}")
            world.worldbody.append(sphere)
            self.objects.append(sphere)

        super().__init__(world, robots=robots, controller_configs=controller_configs,
                         has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
                         use_camera_obs=use_camera_obs, reward_scale=reward_scale)

        self._setup_references()

    def _setup_references(self):
        super()._setup_references()
        # 获取末端执行器的site ID，用于计算与物体的距离
        self.ee_site_id = self.sim.model.site_name2id(self.robot.gripper.important_sites["grip_site"])

        # 为每个物体记录其主体ID
        self.object_body_ids = [self.sim.model.body_name2id(obj.attrib['name']) for obj in self.objects]

    def reset(self):
        obs = super().reset()
        # 随机重新放置物体位置
        for i, obj in enumerate(self.objects):
            pos = np.random.uniform(low=[0.75, -0.2, 0.8], high=[0.85, 0.2, 0.8])
            self.sim.model.body_pos[self.object_body_ids[i]] = pos
        self.sim.forward()
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)
        reward = self.compute_reward()

        return obs, reward, done, info

    def compute_reward(self):
        # 计算奖励函数

        # 1. 获取末端执行器位置
        ee_pos = self.sim.data.site_xpos[self.ee_site_id]

        # 2. 计算与最近球体的距离
        dists = []
        for body_id in self.object_body_ids:
            obj_pos = self.sim.data.body_xpos[body_id]
            dist = np.linalg.norm(ee_pos - obj_pos)
            dists.append(dist)
        min_dist = np.min(dists) if len(dists) > 0 else 0.0

        # 距离奖励（距离越近，奖励越高）
        # 使用 -min_dist 作为基础奖励，让机器人趋近物体。可以根据任务需要缩放。
        distance_reward = -min_dist

        # 3. 判断是否抓取成功
        # 简单判定方式：如果物体离开桌面一定高度（> 0.82, 即离桌面2cm以上）且在夹爪附近，则认为抓取成功。
        grasp_reward = 0.0
        lift_reward = 0.0

        gripper_pos = ee_pos
        gripper_closed = self._gripper_closed()
        # 遍历物体，寻找是否有被夹住的物体
        for body_id in self.object_body_ids:
            obj_pos = self.sim.data.body_xpos[body_id]
            # 如果物体在夹爪非常近的范围内（例如 0.05米内）并且物体比桌面高出一定距离，我们认为已经抓起。
            if np.linalg.norm(gripper_pos - obj_pos) < 0.05 and gripper_closed and obj_pos[
                2] > self.table_offset + 0.02:
                # 抓取奖励
                grasp_reward = 1.0
                # 抬升奖励（越高奖励越多）
                lift_reward = obj_pos[2] - self.table_offset
                break

        # 总奖励
        reward = distance_reward + grasp_reward + lift_reward

        return reward

    def _gripper_closed(self):
        # 简单判定夹爪是否闭合：通过查看夹爪两个指尖之间的距离
        # 在robosuite中，gripper有相应的关节状态可以获取
        left_finger_qpos = self.sim.data.qpos[self.sim.model.joint_name2id(self.robot.gripper.joints[0])]
        right_finger_qpos = self.sim.data.qpos[self.sim.model.joint_name2id(self.robot.gripper.joints[1])]
        # 当指尖接近时，说明夹爪闭合
        # 根据 PandaGripper 的特性，它的指爪在0处是闭合，打开是正值
        # 这里当左右指爪接近时认为抓取动作执行
        if (abs(left_finger_qpos - right_finger_qpos) < 0.02):
            return True
        return False