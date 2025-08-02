#!/usr/bin/env python3
"""
Example usage of the UnitreeG1PlaceAppleInBowlStandalone-v1 task
"""

import mani_skill
from mani_skill.utils.registration import make
import numpy as np

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Create the environment
    env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
               control_mode="pd_joint_delta_pos", 
               obs_mode="rgbd", 
               reward_mode="dense",
               render_mode="human")
    
    # Reset the environment
    obs, info = env.reset()
    print(f"Environment reset. Observation keys: {list(obs.keys())}")
    
    # Run a simple episode
    total_reward = 0
    for step in range(50):
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward.item()
        
        if step % 10 == 0:
            print(f"Step {step}: Reward = {reward.item():.3f}, Success = {info.get('success', False)}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()


def example_with_state_observations():
    """Example using state observations"""
    print("\n=== State Observations Example ===")
    
    # Create environment with state observations
    env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
               control_mode="pd_joint_delta_pos", 
               obs_mode="state", 
               reward_mode="dense")
    
    obs, info = env.reset()
    print(f"State observation keys: {list(obs.keys())}")
    
    # Access state information
    if 'extra' in obs and 'obj_pose' in obs['extra']:
        apple_pos = obs['extra']['obj_pose'][0, :3]  # First 3 elements are position
        bowl_pos = obs['extra']['bowl_pos'][0]  # Bowl position
        print(f"Apple position: {apple_pos}")
        print(f"Bowl position: {bowl_pos}")
    
    env.close()


def example_custom_actions():
    """Example with custom action sequence"""
    print("\n=== Custom Actions Example ===")
    
    env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
               control_mode="pd_joint_delta_pos", 
               obs_mode="rgbd", 
               reward_mode="dense",
               render_mode="human")
    
    obs, info = env.reset()
    
    # Define a simple action sequence (small movements)
    actions = [
        np.array([0.1, 0, 0, 0, 0, 0, 0] + [0] * 18),  # Move right arm forward
        np.array([0, 0.1, 0, 0, 0, 0, 0] + [0] * 18),  # Move right arm up
        np.array([0, 0, 0.1, 0, 0, 0, 0] + [0] * 18),  # Move right arm down
        np.array([0, 0, 0, 0.1, 0, 0, 0] + [0] * 18),  # Rotate right arm
    ]
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Custom action {i+1}: Reward = {reward.item():.3f}")
        
        if terminated or truncated:
            break
    
    env.close()


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_state_observations()
    example_custom_actions()
    
    print("\n=== All examples completed ===") 