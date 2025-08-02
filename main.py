#!/usr/bin/env python3
"""
Standalone UnitreeG1PlaceAppleInBowl-v1 task implementation
Based on ManiSkill custom task tutorial
"""

import copy
import os
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import (
    UnitreeG1UpperBodyWithHeadCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


class HumanoidPickPlaceEnv(BaseEnv):
    """Base class for humanoid pick and place tasks"""
    
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    kitchen_scene_scale = 1.0

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Remove max_episode_steps from kwargs as it's handled by the decorator
        kwargs.pop('max_episode_steps', None)
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.scene_builder.initialize(env_idx)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()


class HumanoidPlaceAppleInBowl(HumanoidPickPlaceEnv):
    """Apple in bowl placement task with dense rewards"""
    
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    @property
    def _default_sensor_configs(self):
        return CameraConfig(
            "base_camera",
            sapien.Pose(
                [0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            128,
            128,
            np.pi / 2,
            0.01,
            100,
        )

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien.Pose(
                [0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            512,
            512,
            np.pi / 2,
            0.01,
            100,
        )

    def _load_scene(self, options: Dict):
        """Load the scene with apple and bowl objects"""
        super()._load_scene(options)
        scale = self.kitchen_scene_scale
        
        # Load the bowl (target container)
        builder = self.scene.create_actor_builder()
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        
        # Use relative path for assets - in a real implementation you'd copy assets
        model_dir = "/home/cnboonhan/workspaces/humanoid/.venv/lib/python3.11/site-packages/mani_skill/envs/tasks/humanoid/assets"
        
        builder.add_nonconvex_collision_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
            scale=[scale] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.753])
        self.bowl = builder.build_kinematic(name="bowl")

        # Load the apple (object to be picked)
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=os.path.join(model_dir, "apple_1.ply"),
            pose=fix_rotation_pose,
            scale=[scale * 0.8] * 3,  # Scale down to make apple graspable
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "apple_1.glb"),
            scale=[scale * 0.8] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.78])
        self.apple = builder.build(name="apple")

    def evaluate(self):
        """Evaluate success/failure conditions"""
        # Check if apple is within bowl area
        is_obj_placed = (
            torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1) <= 0.05
        )
        
        # Check if robot hand is above bowl and outside it
        hand_outside_bowl = (
            self.agent.right_tcp.pose.p[:, 2] > self.bowl.pose.p[:, 2] + 0.125
        )
        
        # Check if robot is grasping the apple
        is_grasped = self.agent.right_hand_is_grasping(self.apple, max_angle=110)
        
        return {
            "success": is_obj_placed & hand_outside_bowl,
            "hand_outside_bowl": hand_outside_bowl,
            "is_grasped": is_grasped,
        }

    def _get_obs_extra(self, info: Dict):
        """Add task-specific observations"""
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.right_tcp.pose.raw_pose,
        )
        
        if self.obs_mode_struct.use_state:
            obs.update(
                bowl_pos=self.bowl.pose.p,
                obj_pose=self.apple.pose.raw_pose,
                tcp_to_obj_pos=self.apple.pose.p - self.agent.right_tcp.pose.p,
                obj_to_goal_pos=self.bowl.pose.p - self.apple.pose.p,
            )
        return obs

    def _grasp_release_reward(self):
        """Reward for releasing grasp when object is above target"""
        return 1 - torch.tanh(self.agent.right_hand_dist_to_open_grasp())

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward for the task"""
        # Reward for reaching the object
        tcp_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.right_tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # Reward for grasping the object
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # Encourage bringing apple to above the bowl then dropping it
        obj_to_goal_dist = torch.linalg.norm(
            (self.bowl.pose.p + torch.tensor([0, 0, 0.15], device=self.device))
            - self.apple.pose.p,
            axis=1,
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # Once above the goal, encourage hand to stay above bowl and release grasp
        obj_high_above_bowl = obj_to_goal_dist < 0.025
        grasp_release_reward = self._grasp_release_reward()
        reward[obj_high_above_bowl] = (
            4
            + place_reward[obj_high_above_bowl]
            + grasp_release_reward[obj_high_above_bowl]
        )
        reward[info["success"]] = (
            8 + (place_reward + grasp_release_reward)[info["success"]]
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalized version of dense reward"""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10


@register_env("UnitreeG1PlaceAppleInBowlStandalone-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlStandaloneEnv(HumanoidPlaceAppleInBowl):
    """
    **Task Description:**
    Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side

    **Randomizations:**
    - the bowl's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
    - the apple's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
    - the apple's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the apple position is within 0.05m euclidean distance of the bowl's position.
    - the robot's right hand is kept outside the bowl and is above it by at least 0.125m.

    **Goal Specification:**
    - The bowl's 3D position
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1PlaceAppleInBowl-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera
    kitchen_scene_scale = 0.82

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.3, 0, 0.755]
        # Remove max_episode_steps from kwargs as it's handled by the decorator
        kwargs.pop('max_episode_steps', None)
        super().__init__(
            *args,
            robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            scene_config=SceneConfig(contact_offset=0.01),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        """Initialize episode with randomized object positions"""
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            
            # Initialize the robot
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

            # Initialize the apple to be within reach
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            xyz[:, 2] = 0.7335
            self.apple.set_pose(Pose.create_from_pq(xyz, qs))

            # Initialize the bowl
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            xyz[:, :2] += torch.tensor([0.0, -0.4])
            xyz[:, 2] = 0.753
            self.bowl.set_pose(Pose.create_from_pq(xyz))


def test_environment():
    """Test the environment to ensure it works correctly"""
    import mani_skill
    from mani_skill.utils.registration import make
    
    print("Testing UnitreeG1PlaceAppleInBowlStandalone-v1 environment...")
    
    try:
        # Create the environment with human render mode
        env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
                   control_mode="pd_joint_delta_pos", 
                   obs_mode="rgbd", 
                   reward_mode="dense",
                   render_mode="human",
                   max_episode_steps=100)
        
        print("✓ Environment created successfully!")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Reset the environment
        obs, info = env.reset()
        print(f"✓ Environment reset successfully!")
        print(f"  Initial observation keys: {list(obs.keys())}")
        
        print("Starting interactive visualization...")
        print("Press 'q' to quit, or let the episode run for 100 steps")
        
        # Interactive visualization loop
        for step in range(100):
            # Render the environment
            env.render()
            
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:  # Print every 10 steps
                print(f"  Step {step}: Reward = {reward}, Terminated = {terminated}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        env.close()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_environment_headless():
    """Test the environment without visualization for faster testing"""
    import mani_skill
    from mani_skill.utils.registration import make
    
    print("Testing UnitreeG1PlaceAppleInBowlStandalone-v1 environment (headless)...")
    
    try:
        # Create the environment without render mode
        env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
                   control_mode="pd_joint_delta_pos", 
                   obs_mode="rgbd", 
                   reward_mode="dense",
                   max_episode_steps=100)
        
        print("✓ Environment created successfully!")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Reset the environment
        obs, info = env.reset()
        print(f"✓ Environment reset successfully!")
        print(f"  Initial observation keys: {list(obs.keys())}")
        
        # Take a few random actions
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {step + 1}: Reward = {reward}, Terminated = {terminated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def run_interactive_environment():
    """Run the environment with interactive visualization and manual control"""
    import mani_skill
    from mani_skill.utils.registration import make
    import time
    
    print("Starting interactive UnitreeG1PlaceAppleInBowlStandalone-v1 environment...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset")
    print("  - Press 'space' to pause/unpause")
    
    try:
        # Create the environment with human render mode
        env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
                   control_mode="pd_joint_delta_pos", 
                   obs_mode="rgbd", 
                   reward_mode="dense",
                   render_mode="human",
                   max_episode_steps=100)
        
        # Reset the environment
        obs, info = env.reset()
        
        # Interactive loop
        paused = False
        step = 0
        
        while step < 100:
            if not paused:
                # Take a random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if step % 10 == 0:
                    print(f"Step {step}: Reward = {reward}, Success = {info.get('success', False)}")
                
                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    obs, info = env.reset()
                    step = 0
                    continue
                
                step += 1
            
            # Render the environment
            env.render()
            time.sleep(0.05)  # 20 FPS
        
        env.close()
        print("✓ Interactive session completed!")
        
    except Exception as e:
        print(f"✗ Error during interactive session: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "headless":
            test_environment_headless()
        elif mode == "interactive":
            run_interactive_environment()
        else:
            print("Usage: python main.py [headless|interactive]")
            print("  headless: Run without visualization (faster)")
            print("  interactive: Run with visualization and manual control")
            print("  (no args): Run with visualization and random actions")
    else:
        test_environment()

