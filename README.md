# Humanoid - Custom ManiSkill Task

This project contains a standalone implementation of the UnitreeG1PlaceAppleInBowl-v1 task based on the [ManiSkill custom task tutorial](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html).

## Task Description

**UnitreeG1PlaceAppleInBowlStandalone-v1**: Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side.

### Key Features
- **Standalone Implementation**: Complete task implementation in a single file
- **Human Visualization**: Interactive 3D visualization with render mode
- **Multiple Observation Modes**: RGBD camera observations and state observations
- **Dense Rewards**: Sophisticated reward shaping for learning
- **Randomization**: Object positions are randomized for generalization

### Success Conditions
- The apple position is within 0.05m euclidean distance of the bowl's position
- The robot's right hand is kept outside the bowl and is above it by at least 0.125m

### Randomizations
- Bowl's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]
- Apple's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]
- Apple's z-axis rotation is randomized to a random angle

## Files

- `main.py`: Complete standalone task implementation with visualization
- `example_usage.py`: Examples showing different ways to use the environment
- `README.md`: This documentation

## Usage

### Basic Visualization
```bash
# Run with human render mode (default)
uv run python main.py

# Run headless (faster, no visualization)
uv run python main.py headless

# Run interactive mode with manual control
uv run python main.py interactive
```

### Programmatic Usage
```python
import mani_skill
from mani_skill.utils.registration import make

# Create environment
env = make("UnitreeG1PlaceAppleInBowlStandalone-v1", 
           control_mode="pd_joint_delta_pos", 
           obs_mode="rgbd", 
           reward_mode="dense",
           render_mode="human")

# Reset and run
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Example Scripts
```bash
# Run the example usage script
uv run python example_usage.py
```

## Custom Task Structure

Based on the [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html), this implementation includes all required components:

### 1. Task Class Setup
```python
@register_env("UnitreeG1PlaceAppleInBowlStandalone-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlStandaloneEnv(HumanoidPlaceAppleInBowl):
```

### 2. Loading (Robots, Assets, Sensors)
```python
def _load_scene(self, options: Dict):
    # Load bowl and apple objects
    # Set up cameras and sensors
```

### 3. Episode Initialization / Randomization
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
    # Randomize object positions
    # Initialize robot pose
```

### 4. Success/Failure Conditions
```python
def evaluate(self):
    # Check if apple is in bowl
    # Check if hand is above bowl
    return {"success": success_condition}
```

### 5. Extra Observations
```python
def _get_obs_extra(self, info: Dict):
    # Add task-specific observations
    # Include object positions, grasping state
```

### 6. Dense Reward Function
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    # Reward for reaching, grasping, placing
    # Encourage proper hand positioning
```

### 7. Camera/Sensor Setup
```python
@property
def _default_sensor_configs(self):
    # Configure cameras for observations
    # Set up render cameras for visualization
```

## Environment Modes

### Control Modes
- `pd_joint_delta_pos`: Joint position control with delta actions
- `pd_joint_pos`: Direct joint position control

### Observation Modes
- `rgbd`: RGB and depth camera observations
- `state`: Ground truth state information
- `rgbd+state`: Combined visual and state observations

### Reward Modes
- `dense`: Shaped rewards for learning
- `sparse`: Binary success/failure rewards
- `normalized_dense`: Normalized dense rewards

## Customization

To modify this task for different objects or scenarios:

1. **Change Objects**: Modify the `_load_scene()` method to load different objects
2. **Adjust Randomization**: Modify position ranges in `_initialize_episode()`
3. **Update Success Conditions**: Modify the `evaluate()` method
4. **Customize Rewards**: Modify the `compute_dense_reward()` method
5. **Add New Observations**: Extend the `_get_obs_extra()` method

## Dependencies

- ManiSkill 3.0.0b21
- PyTorch
- SAPIEN
- NumPy

## Troubleshooting

### Common Issues
1. **Asset Loading**: Ensure the assets directory is accessible
2. **GPU Memory**: Adjust `max_rigid_contact_count` in sim config for GPU memory issues
3. **Render Mode**: Use `headless` mode if visualization doesn't work

### Performance Tips
- Use `obs_mode="state"` for faster training (no rendering)
- Use `render_mode=None` for headless operation
- Adjust camera resolution for speed vs quality trade-off

This implementation provides a complete, standalone example of how to create custom tasks in ManiSkill following the official documentation structure.