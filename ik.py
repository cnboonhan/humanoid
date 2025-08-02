# python3 ik.py --path .venv/lib/python3.11/site-packages/mani_skill/assets/robots/g1_humanoid/g1.urdf  --port 8080

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


# Global variables to store robot state
current_robot_config = None
actuated_joint_names = None

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