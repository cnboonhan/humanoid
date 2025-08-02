# python3 ik_keyboard.py --path .venv/lib/python3.11/site-packages/mani_skill/assets/robots/g1_humanoid/g1.urdf  --port 8080 --flask-port 5000
# curl http://localhost:5000/config

import time
import tyro
import viser
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
from yourdfpy import URDF
from _solve_ik_with_multiple_targets import solve_ik_with_multiple_targets
from flask import Flask, jsonify
import threading
from pynput import keyboard
import asyncio
from vuer import Vuer, VuerSession
from vuer.schemas import Hands


# Global variables to store robot state
current_robot_config = None
actuated_joint_names = None

# Keyboard control variables
keyboard_step_size = 0.01  # meters per key press
keyboard_pressed = set()
gripper_step_size = 0.1  # radians per button press

# Create Flask app
app = Flask(__name__)

@app.route('/config', methods=['GET'])
def get_robot_config():
    """Return the current robot configuration as JSON"""
    global current_robot_config, actuated_joint_names
    
    if current_robot_config is None or actuated_joint_names is None:
        return jsonify({"error": "Robot not initialized"}), 500
    
    # Create a dictionary mapping joint names to their current values
    config_dict = {
        "joint_config": dict(zip(actuated_joint_names, current_robot_config.tolist())),
        "timestamp": time.time()
    }
    
    return jsonify(config_dict)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

def run_flask_server(flask_port=5000):
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

def on_key_press(key):
    """Handle key press events"""
    global keyboard_pressed
    try:
        # Convert pynput key to string representation
        if hasattr(key, 'char') and key.char:
            keyboard_pressed.add(key.char)
        elif hasattr(key, 'name'):
            keyboard_pressed.add(key.name)
        else:
            # Handle special keys
            if key == keyboard.Key.up:
                keyboard_pressed.add('up')
            elif key == keyboard.Key.down:
                keyboard_pressed.add('down')
            elif key == keyboard.Key.left:
                keyboard_pressed.add('left')
            elif key == keyboard.Key.right:
                keyboard_pressed.add('right')
    except AttributeError:
        pass

def on_key_release(key):
    """Handle key release events"""
    global keyboard_pressed
    try:
        # Convert pynput key to string representation
        if hasattr(key, 'char') and key.char:
            keyboard_pressed.discard(key.char)
        elif hasattr(key, 'name'):
            keyboard_pressed.discard(key.name)
        else:
            # Handle special keys
            if key == keyboard.Key.up:
                keyboard_pressed.discard('up')
            elif key == keyboard.Key.down:
                keyboard_pressed.discard('down')
            elif key == keyboard.Key.left:
                keyboard_pressed.discard('left')
            elif key == keyboard.Key.right:
                keyboard_pressed.discard('right')
    except AttributeError:
        pass

def update_target_positions(ik_target_0, ik_target_1, server):
    """Update target positions based on keyboard input"""
    global keyboard_pressed, keyboard_step_size
    
    # Track if any position changed
    position_changed = False
    new_pos_0 = np.array(ik_target_0.position)
    new_pos_1 = np.array(ik_target_1.position)
    
    # Right hand controls (TFGH + RY for Z-axis)
    if 'f' in keyboard_pressed:  # Forward (Y+)
        new_pos_1[1] += keyboard_step_size
        position_changed = True
    if 'h' in keyboard_pressed:  # Backward (Y-)
        new_pos_1[1] -= keyboard_step_size
        position_changed = True
    if 'g' in keyboard_pressed:  # Left (X-)
        new_pos_1[0] -= keyboard_step_size
        position_changed = True
    if 't' in keyboard_pressed:  # Right (X+)
        new_pos_1[0] += keyboard_step_size
        position_changed = True
    if 'r' in keyboard_pressed:  # Up (Z+)
        new_pos_1[2] += keyboard_step_size
        position_changed = True
    if 'y' in keyboard_pressed:  # Down (Z-)
        new_pos_1[2] -= keyboard_step_size
        position_changed = True
    
    # Left hand controls (IJKL + UO for Z-axis)
    if 'j' in keyboard_pressed:  # Forward (Y+)
        new_pos_0[1] += keyboard_step_size
        position_changed = True
    if 'l' in keyboard_pressed:  # Backward (Y-)
        new_pos_0[1] -= keyboard_step_size
        position_changed = True
    if 'k' in keyboard_pressed:  # Left (X-)
        new_pos_0[0] -= keyboard_step_size
        position_changed = True
    if 'i' in keyboard_pressed:  # Right (X+)
        new_pos_0[0] += keyboard_step_size
        position_changed = True
    if 'u' in keyboard_pressed:  # Up (Z+)
        new_pos_0[2] += keyboard_step_size
        position_changed = True
    if 'o' in keyboard_pressed:  # Down (Z-)
        new_pos_0[2] -= keyboard_step_size
        position_changed = True
    
    # If any position changed, update the transform controls
    if position_changed:
        # Update the transform control positions
        ik_target_0.position = new_pos_0
        ik_target_1.position = new_pos_1

def main(path: str, port: int, flask_port: int = 5000) -> None:
    global current_robot_config, actuated_joint_names
    
    urdf = URDF.load(path, load_collision_meshes=True, build_collision_scene_graph=True)
    
    robot = pk.Robot.from_urdf(urdf)
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    
    print(f"Robot name: {urdf.robot}")
    print(f"Number of joints: {len(urdf.joint_map)}")
    print(f"Number of links: {len(urdf.link_map)}")
    
    actuated_joints = [name for name, joint in urdf.joint_map.items() if joint.type != 'fixed']
    actuated_joint_names = actuated_joints  # Store globally for Flask API
    print(f"Actuated joints ({len(actuated_joints)}): {actuated_joints}")
    
    initial_config = []
    for joint_name, (lower, upper) in urdf_vis.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        initial_config.append(initial_pos)
    
    initial_config_array = np.array(initial_config)
    current_robot_config = initial_config_array.copy()  # Initialize global config
    urdf_vis.update_cfg(initial_config_array)
    print(f"Initial configuration set with {len(initial_config)} joints")
    
    target_link_names = ["right_palm_link", "left_palm_link"]
    
    link_names = list(urdf.link_map.keys())
    right_palm_idx = link_names.index("right_palm_link")
    left_palm_idx = link_names.index("left_palm_link")
    
    print(f"Right palm link index: {right_palm_idx}")
    print(f"Left palm link index: {left_palm_idx}")
    
    right_palm_transform = robot.forward_kinematics(initial_config_array, right_palm_idx)
    left_palm_transform = robot.forward_kinematics(initial_config_array, left_palm_idx)
    
    # The forward kinematics returns a matrix where each row is a link transform
    # Extract the position from the specific link row (last 3 columns)
    right_palm_pos = np.array(right_palm_transform[right_palm_idx, -3:])
    left_palm_pos = np.array(left_palm_transform[left_palm_idx, -3:])
    
    # For now, use default quaternions
    right_palm_quat = (1.0, 0.0, 0.0, 0.0)
    left_palm_quat = (1.0, 0.0, 0.0, 0.0)
    
    print(f"Right palm transform shape: {right_palm_transform.shape}")
    print(f"Right palm initial position: {right_palm_pos}")
    print(f"Left palm initial position: {left_palm_pos}")
    
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=right_palm_pos, wxyz=right_palm_quat
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=left_palm_pos, wxyz=left_palm_quat
    )
    
    # Add some GUI controls
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add reset button
    reset_button = server.gui.add_button("Reset to Initial Pose")
    
    @reset_button.on_click
    def _(_):
        urdf_vis.update_cfg(initial_config_array)
        print("Reset to initial pose")
    
    # Add hand joint control sliders
    server.gui.add_markdown("### Hand Joint Controls")
    
    right_gripper_slider = server.gui.add_slider("Right Hand Joints", 0.0, 1.0, 0.1, 0.5)
    left_gripper_slider = server.gui.add_slider("Left Hand Joints", 0.0, 1.0, 0.1, 0.5)
    
    @right_gripper_slider.on_update
    def _(_):
        global current_robot_config, actuated_joint_names
        new_config = current_robot_config.copy()
        gripper_value = right_gripper_slider.value
        joint_limits = urdf_vis.get_actuated_joint_limits()
        for i, joint_name in enumerate(actuated_joint_names):
            if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']:
                # Get actual joint limits
                lower, upper = joint_limits[joint_name]
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                # Reverse the entire right hand direction
                reversed_value = 1.0 - gripper_value
                # For thumb joints (zero, one, two), reverse the direction again (double reverse = normal)
                if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint']:
                    # Use normal value for thumb joints since we already reversed the whole hand
                    new_config[i] = lower + gripper_value * (upper - lower)
                else:
                    # Use reversed value for other finger joints
                    new_config[i] = lower + reversed_value * (upper - lower)
        current_robot_config = new_config.copy()
        urdf_vis.update_cfg(new_config)
    
    @left_gripper_slider.on_update
    def _(_):
        global current_robot_config, actuated_joint_names
        new_config = current_robot_config.copy()
        gripper_value = left_gripper_slider.value
        joint_limits = urdf_vis.get_actuated_joint_limits()
        for i, joint_name in enumerate(actuated_joint_names):
            if joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint']:
                # Get actual joint limits
                lower, upper = joint_limits[joint_name]
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                # For thumb joints (zero, one, two), reverse the direction
                if joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint']:
                    # Reverse the slider value for thumb joints
                    reversed_value = 1.0 - gripper_value
                    new_config[i] = lower + reversed_value * (upper - lower)
                else:
                    # Map slider value (0-1) to joint position within actual limits
                    new_config[i] = lower + gripper_value * (upper - lower)
        current_robot_config = new_config.copy()
        urdf_vis.update_cfg(new_config)
    
    # Add keyboard step size control
    step_size_handle = server.gui.add_slider("Keyboard Step Size (m)", 0.001, 0.1, 0.01, 0.01)
    
    @step_size_handle.on_update
    def _(_):
        global keyboard_step_size
        keyboard_step_size = step_size_handle.value
    
    # Add keyboard controls info
    server.gui.add_markdown("""
    ## Keyboard Controls
    
    **Right Hand (Red Target):**
    - T/G: Forward/Backward (Y-axis)
    - F/H: Left/Right (X-axis)
    - R/Y: Up/Down (Z-axis)
    
    **Left Hand (Blue Target):**
    - I/K: Forward/Backward (Y-axis)
    - J/L: Left/Right (X-axis)
    - U/O: Up/Down (Z-axis)
    
    **General:**
    - Adjust step size using the slider above
    - Use the transform controls for precise positioning
    - Use the hand joint sliders above to control all hand joints
    """)
    
    # Set up keyboard event listeners
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.start()
    
    print("Keyboard controls enabled:")
    print("Right hand: TFGH (XY) + RY (Z)")
    print("Left hand: IJKL (XY) + UO (Z)")
    print("Hand joint controls: Use GUI sliders")
    print("Adjust step size with the GUI slider")
    
    # Define upper body joint indices (arms and hands only)
    # Based on the actuated joints list: upper body starts from index 12 (torso_joint) onwards
    # Lower body: indices 0-11 (left_hip_pitch_joint to right_ankle_roll_joint)
    # Upper body: indices 12-36 (torso_joint to right_six_joint)
    lower_body_indices = list(range(12))  # 0-11: legs and hips
    upper_body_indices = list(range(12, len(initial_config)))  # 12-36: torso, arms, hands
    
    print(f"Lower body joint indices: {lower_body_indices}")
    print(f"Upper body joint indices: {upper_body_indices}")
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, args=(flask_port,), daemon=True)
    flask_thread.start()
    print(f"Flask server started on port {flask_port}")
    print(f"Robot config available at: http://localhost:{flask_port}/config")
    print(f"Health check available at: http://localhost:{flask_port}/health")
    
    while True:
        # Update target positions based on keyboard input
        update_target_positions(ik_target_0, ik_target_1, server)
        
        # Solve IK for both targets
        start_time = time.time()
        try:
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([ik_target_0.position, ik_target_1.position]),
                target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            )
            
            # Create a new configuration that only updates upper body joints
            # Keep lower body joints at their initial values
            current_config = initial_config_array.copy()
            current_config[upper_body_indices] = solution[upper_body_indices]
            
            # Preserve hand joint changes from sliders
            for i, joint_name in enumerate(actuated_joint_names):
                if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint', 
                                 'left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint']:
                    # Keep the hand joint values from the current robot config (set by sliders)
                    current_config[i] = current_robot_config[i]
            
            # Store current configuration globally for Flask API
            current_robot_config = current_config.copy()
            
            urdf_vis.update_cfg(current_config)
            
        except Exception as e:
            print(f"IK solver failed: {e}")
        
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)