# python3 ik.py --path .venv/lib/python3.11/site-packages/mani_skill/assets/robots/g1_humanoid/g1.urdf  --port 8080 --flask-port 5000
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
from vuer import Vuer, VuerSession
from vuer.schemas import Hands
from asyncio import sleep

# Check if SSL certificates exist
import os
cert_path = "public.crt"
key_path = "private.key"

if not os.path.exists(cert_path) or not os.path.exists(key_path):
    print(f"WARNING: SSL certificates not found!")
    print(f"  Certificate: {cert_path} - {'EXISTS' if os.path.exists(cert_path) else 'MISSING'}")
    print(f"  Private key: {key_path} - {'EXISTS' if os.path.exists(key_path) else 'MISSING'}")
    print("Hand tracking may not work without SSL certificates.")
    # Try without SSL
    vuer_app = Vuer(host="0.0.0.0", port=8012)
else:
    print(f"SSL certificates found, using secure connection")
    vuer_app = Vuer(host="0.0.0.0", port=8012, cert=cert_path, key=key_path)


# Global variables to store robot state
current_robot_config = None
actuated_joint_names = None

# Global variables to store initial hand positions from hand tracking
initial_hand_tracking_left_position = None
initial_hand_tracking_right_position = None

# Global variables to store initial hand orientations from hand tracking
initial_hand_tracking_left_orientation = None
initial_hand_tracking_right_orientation = None

# Global variable for right hand target point (default: origin)
right_hand_target_point = np.array([0.02, 0.2, 0.8])  # Closer to hand level

# Global variable for left hand target point (default: origin)
left_hand_target_point = np.array([0.02, 0.2, 0.8])

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

def extract_hand_position(hand_data, hand_name):
    """
    Extract hand position from the hand tracking data.
    hand_data: Float32Array of 25 * 16 values (25 joints * 16 matrix values)
    hand_name: 'left' or 'right' for logging
    """
    if hand_data is None:
        return None
    
    # Convert to numpy array for easier manipulation
    hand_array = np.array(hand_data)
    
    # Each joint has a 4x4 transform matrix (16 values)
    # The wrist joint is at index 0, which gives us the overall hand position
    wrist_matrix = hand_array[:16].reshape(4, 4)
    
    # In column-major order, the translation is in the last column (indices 12, 13, 14)
    # But numpy reshape creates a row-major matrix, so we need to transpose
    # to get the correct column-major interpretation
    wrist_matrix = wrist_matrix.T  # Transpose to get column-major interpretation
    
    # Extract position from the last column of the transform matrix
    position = wrist_matrix[:3, 3]
    
    return position

def extract_hand_orientation(hand_data, hand_name):
    """
    Extract hand orientation (roll, pitch, yaw) from the hand tracking data.
    hand_data: Float32Array of 25 * 16 values (25 joints * 16 matrix values)
    hand_name: 'left' or 'right' for logging
    
    Returns: (roll_deg, pitch_deg, yaw_deg) in degrees
    """
    if hand_data is None:
        return None, None, None
    
    try:
        # Convert to numpy array for easier manipulation
        hand_array = np.array(hand_data)
        
        # Each joint has a 4x4 transform matrix (16 values)
        # The wrist joint is at index 0, which gives us the overall hand orientation
        wrist_matrix = hand_array[:16].reshape(4, 4)
        
        # In column-major order, the translation is in the last column (indices 12, 13, 14)
        # But numpy reshape creates a row-major matrix, so we need to transpose
        # to get the correct column-major interpretation
        wrist_matrix = wrist_matrix.T  # Transpose to get column-major interpretation
        
        # Extract rotation matrix (3x3 upper left part)
        rotation_matrix = wrist_matrix[:3, :3]
        
        # Convert rotation matrix to euler angles
        # Using a simple method to extract roll, pitch, yaw from rotation matrix
        # This assumes XYZ rotation order
        
        # Extract roll (rotation around X-axis)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        # Extract pitch (rotation around Y-axis)
        pitch = np.arctan2(-rotation_matrix[2, 0], 
                          np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        
        # Extract yaw (rotation around Z-axis)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Convert to degrees
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        print(f"{hand_name.capitalize()} hand orientation: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°")
        
        # Debug: Check for unusual values
        if abs(roll_deg) > 180 or abs(pitch_deg) > 180 or abs(yaw_deg) > 180:
            print(f"WARNING: {hand_name} has unusual orientation values!")
            print(f"Rotation matrix: {rotation_matrix}")
        
        return roll_deg, pitch_deg, yaw_deg
        
    except Exception as e:
        print(f"Error extracting {hand_name} hand orientation: {e}")
        return None, None, None

def calculate_palm_orientation_to_origin(hand_position, target_point=None):
    """
    Calculate palm orientation to face a target point.
    
    Args:
        hand_position: Current hand position [x, y, z]
        target_point: Target point to face (uses global right_hand_target_point if None)
    
    Returns:
        quaternion: Quaternion representing palm orientation to face target
    """
    global right_hand_target_point
    if target_point is None:
        target_point = right_hand_target_point
    if hand_position is None:
        return np.array([1, 0, 0, 0])  # Default identity quaternion
    
    # Calculate direction vector from target to hand (reversed)
    direction = hand_position - target_point
    distance = np.linalg.norm(direction)
    
    print(f"Hand position: {hand_position}, target point: {target_point}")
    print(f"Direction vector: {direction}, distance: {distance}")
    
    if distance < 0.001:  # Hand is very close to target
        print("Hand too close to target, using identity quaternion")
        return np.array([1, 0, 0, 0])  # Default identity quaternion
    
    # Normalize direction vector
    direction = direction / distance
    
    # Calculate rotation to align palm normal (assumed to be -Z axis) with direction
    # We want the palm to face the target, so we need to rotate the palm normal to point towards the target
    
    # Try different palm normal directions
    palm_normal = np.array([0, -1, 0])   # Palm normal pointing down (negated)
    
    print(f"Using palm normal: {palm_normal}")
    
    # Cross product to find rotation axis
    rotation_axis = np.cross(palm_normal, direction)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 0.001:  # Vectors are parallel or anti-parallel
        if np.dot(palm_normal, direction) > 0:
            return np.array([1, 0, 0, 0])  # Already aligned
        else:
            # Anti-parallel, rotate 180 degrees around X axis
            return np.array([0, 1, 0, 0])
    
    # Normalize rotation axis
    rotation_axis = rotation_axis / axis_norm
    
    # Calculate rotation angle
    cos_angle = np.dot(palm_normal, direction)
    cos_angle = np.clip(cos_angle, -1, 1)  # Clamp to valid range
    angle = np.arccos(cos_angle)
    
    # Convert to quaternion
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    
    quaternion = np.array([
        cos_half_angle,
        rotation_axis[0] * sin_half_angle,
        rotation_axis[1] * sin_half_angle,
        rotation_axis[2] * sin_half_angle
    ])
    
    print(f"Rotation axis: {rotation_axis}, angle: {np.degrees(angle):.1f}°")
    print(f"Calculated quaternion: {quaternion}")
    
    return quaternion

@vuer_app.add_handler("HAND_MOVE")
async def handler(event, session):
    """Handle hand movement events and extract hand positions"""
    hand_data = event.value
    
    if hand_data is not None:
        global initial_hand_tracking_left_position, initial_hand_tracking_right_position, initial_hand_tracking_left_orientation, initial_hand_tracking_right_orientation
        
        if len(hand_data['left']) > 2:
            try:
                left_value = hand_data['left']
                left_position = extract_hand_position(left_value, 'left')
                
                # Extract left hand orientation
                left_roll, left_pitch, left_yaw = extract_hand_orientation(left_value, 'left')
                
                # Capture initial left hand position and orientation if not already captured
                if initial_hand_tracking_left_position is None and left_position is not None:
                    initial_hand_tracking_left_position = left_position.copy()
                
                if initial_hand_tracking_left_orientation is None and left_roll is not None:
                    initial_hand_tracking_left_orientation = np.array([left_roll, left_pitch, left_yaw])
                    print(f"Initial left hand orientation captured: roll={left_roll:.1f}°, pitch={left_pitch:.1f}°, yaw={left_yaw:.1f}°")
                
                # Calculate and print orientation delta if initial orientation is captured
                if initial_hand_tracking_left_orientation is not None and left_roll is not None:
                    current_orientation = np.array([left_roll, left_pitch, left_yaw])
                    orientation_delta = current_orientation - initial_hand_tracking_left_orientation
                    print(f"Left hand - Current: roll={left_roll:.1f}°, pitch={left_pitch:.1f}°, yaw={left_yaw:.1f}°")
                    print(f"Left hand - Initial: roll={initial_hand_tracking_left_orientation[0]:.1f}°, pitch={initial_hand_tracking_left_orientation[1]:.1f}°, yaw={initial_hand_tracking_left_orientation[2]:.1f}°")
                    print(f"Left hand orientation delta: roll={orientation_delta[0]:+.1f}°, pitch={orientation_delta[1]:+.1f}°, yaw={orientation_delta[2]:+.1f}°")
                    
            except Exception as e:
                print(f"Error accessing 'left': {e}")
                
        if len(hand_data['right']) > 2:
            try:
                right_value = hand_data['right']
                right_position = extract_hand_position(right_value, 'right')
                
                # Extract right hand orientation
                right_roll, right_pitch, right_yaw = extract_hand_orientation(right_value, 'right')
                
                # Capture initial right hand position and orientation if not already captured
                if initial_hand_tracking_right_position is None and right_position is not None:
                    initial_hand_tracking_right_position = right_position.copy()
                
                if initial_hand_tracking_right_orientation is None and right_roll is not None:
                    initial_hand_tracking_right_orientation = np.array([right_roll, right_pitch, right_yaw])
                    print(f"Initial right hand orientation captured: roll={right_roll:.1f}°, pitch={right_pitch:.1f}°, yaw={right_yaw:.1f}°")
                
                # Calculate and print orientation delta if initial orientation is captured
                if initial_hand_tracking_right_orientation is not None and right_roll is not None:
                    current_orientation = np.array([right_roll, right_pitch, right_yaw])
                    orientation_delta = current_orientation - initial_hand_tracking_right_orientation
                    print(f"Right hand - Current: roll={right_roll:.1f}°, pitch={right_pitch:.1f}°, yaw={right_yaw:.1f}°")
                    print(f"Right hand - Initial: roll={initial_hand_tracking_right_orientation[0]:.1f}°, pitch={initial_hand_tracking_right_orientation[1]:.1f}°, yaw={initial_hand_tracking_right_orientation[2]:.1f}°")
                    print(f"Right hand orientation delta: roll={orientation_delta[0]:+.1f}°, pitch={orientation_delta[1]:+.1f}°, yaw={orientation_delta[2]:+.1f}°")
                    
            except Exception as e:
                print(f"Error accessing 'right': {e}")

@vuer_app.add_handler("test")
async def test_handler(event, session):
    """Test handler to verify Vuer is working"""
    print(f"Test event received: {event.key} - {event.value}")

@vuer_app.add_handler("*")
async def debug_all_events(event, session):
    """Debug handler to see all events"""
    print(f"DEBUG EVENT: {event.key} - {type(event.value)} - {event.value}")

def run_flask_server(flask_port=5000):
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

def main(path: str, port: int, flask_port: int = 5000) -> None:
    global current_robot_config, actuated_joint_names, right_gripper_state, left_gripper_state, right_gripper_target, left_gripper_target
    
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
    
    # Add gripper control buttons
    server.gui.add_markdown("### Gripper Controls")
    right_open_button = server.gui.add_button("Right Hand Open")
    right_close_button = server.gui.add_button("Right Hand Close")
    left_open_button = server.gui.add_button("Left Hand Open")
    left_close_button = server.gui.add_button("Left Hand Close")
    
    # Global variables to track gripper states
    right_gripper_state = 0.0  # 0.0 = open, 1.0 = closed
    left_gripper_state = 0.0   # 0.0 = open, 1.0 = closed
    
    # Global variables to track target gripper states for gradual transitions
    right_gripper_target = 0.0
    left_gripper_target = 0.0
    
    # Interpolation speed (lower = slower)
    gripper_lerp_factor = 0.05
    
    @right_open_button.on_click
    def _(_):
        global right_gripper_target
        right_gripper_target = 1.0  # Open (reversed)
        print("Right hand opening...")
    
    @right_close_button.on_click
    def _(_):
        global right_gripper_target
        right_gripper_target = 0.0  # Closed (reversed)
        print("Right hand closing...")
    
    @left_open_button.on_click
    def _(_):
        global left_gripper_target
        left_gripper_target = 1.0  # Open (reversed)
        print("Left hand opening...")
    
    @left_close_button.on_click
    def _(_):
        global left_gripper_target
        left_gripper_target = 0.0  # Closed (reversed)
        print("Left hand closing...")
    
    # Add target point configuration sliders
    server.gui.add_markdown("### Hand Tracking Target Configuration")
    target_x_slider = server.gui.add_slider("Right Hand Target X", -1.0, 1.0, 0.01, 0.02)
    target_y_slider = server.gui.add_slider("Right Hand Target Y", -1.0, 1.0, 0.01, 0.2)
    target_z_slider = server.gui.add_slider("Right Hand Target Z", -1.0, 1.0, 0.01, 0.8)
    
    # Add left hand target configuration sliders
    server.gui.add_markdown("### Left Hand Target Configuration")
    left_target_x_slider = server.gui.add_slider("Left Hand Target X", -1.0, 1.0, 0.01, 0.02)
    left_target_y_slider = server.gui.add_slider("Left Hand Target Y", -1.0, 1.0, 0.01, 0.2)
    left_target_z_slider = server.gui.add_slider("Left Hand Target Z", -1.0, 1.0, 0.01, 0.8)
    
    @target_x_slider.on_update
    def update_target_x(_):
        global right_hand_target_point
        right_hand_target_point[0] = target_x_slider.value
        print(f"Right hand target X updated to: {target_x_slider.value}")
    
    @target_y_slider.on_update
    def update_target_y(_):
        global right_hand_target_point
        right_hand_target_point[1] = target_y_slider.value
        print(f"Right hand target Y updated to: {target_y_slider.value}")
    
    @target_z_slider.on_update
    def update_target_z(_):
        global right_hand_target_point
        right_hand_target_point[2] = target_z_slider.value
        print(f"Right hand target Z updated to: {target_z_slider.value}")
    
    @left_target_x_slider.on_update
    def update_left_target_x(_):
        global left_hand_target_point
        left_hand_target_point[0] = left_target_x_slider.value
        print(f"Left hand target X updated to: {left_target_x_slider.value}")
    
    @left_target_y_slider.on_update
    def update_left_target_y(_):
        global left_hand_target_point
        left_hand_target_point[1] = left_target_y_slider.value
        print(f"Left hand target Y updated to: {left_target_y_slider.value}")
    
    @left_target_z_slider.on_update
    def update_left_target_z(_):
        global left_hand_target_point
        left_hand_target_point[2] = left_target_z_slider.value
        print(f"Left hand target Z updated to: {left_target_z_slider.value}")
    
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
    
    # Start Vuer hand tracking (non-blocking)
    print("Starting Vuer hand tracking...")
    print("Make sure you have SSL certificates set up for hand tracking to work!")
    print("You can access the hand tracking interface at: https://localhost:8080")
    
    # Set up hand tracking session
    @vuer_app.spawn(start=False)
    async def hand_tracking_session(session: VuerSession):
        """Set up hand tracking with Vuer"""
        print("Setting up hand tracking...")
        print(f"Session type: {type(session)}")
        
        try:
            # Add the Hands component to start tracking
            hands_component = Hands(
                stream=True,  # Important: set stream=True to start streaming
                key="hands",
                # hideLeft=False,       # hides the hand, but still streams the data
                # hideRight=False,      # hides the hand, but still streams the data
                # disableLeft=False,    # disables the left data stream, also hides the hand
                # disableRight=False,   # disables the right data stream, also hides the hand
            )
            print(f"Hands component created: {hands_component}")
            
            session.upsert(
                hands_component,
                to="bgChildren",
            )
            
            print("Hand tracking component added. Hand movements will be printed to console.")
            
            # Test if session is working by sending a test message
            await session.add(
                {"type": "test", "message": "Hand tracking session is active"},
                to="bgChildren"
            )
            print("Test message sent to session")
            
        except Exception as e:
            print(f"Error setting up hand tracking: {e}")
            import traceback
            traceback.print_exc()
        
        # Keep the session alive
        while True:
            await sleep(1)
    
    # Start Vuer app in a separate thread with proper event loop (non-blocking)
    def run_vuer_with_event_loop():
        import asyncio
        import traceback
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            print("Vuer thread: Starting Vuer app...")
            vuer_app.run()
        except Exception as e:
            print(f"Vuer thread error: {e}")
            traceback.print_exc()
        finally:
            print("Vuer thread: Closing event loop")
            loop.close()
    
    vuer_thread = threading.Thread(target=run_vuer_with_event_loop, daemon=True)
    vuer_thread.start()
    print("Vuer app started in background thread with event loop")
    
    while True:
        # Solve IK for both targets
        start_time = time.time()
        
        # Gradually interpolate gripper states towards targets
        
        # Interpolate right gripper
        if abs(right_gripper_state - right_gripper_target) > 0.001:
            right_gripper_state += gripper_lerp_factor * (right_gripper_target - right_gripper_state)
        
        # Interpolate left gripper
        if abs(left_gripper_state - left_gripper_target) > 0.001:
            left_gripper_state += gripper_lerp_factor * (left_gripper_target - left_gripper_state)
        
        # Visualize the target points being tracked
        try:
            # Visualize right hand target point
            server.scene.add_point_cloud(
                "/right_target_point",
                points=np.array([right_hand_target_point]),
                colors=np.array([[1, 0, 0]]),  # Red color for right hand
                point_size=0.05
            )
            
            # Visualize left hand target point
            server.scene.add_point_cloud(
                "/left_target_point",
                points=np.array([left_hand_target_point]),
                colors=np.array([[0, 0, 1]]),  # Blue color for left hand
                point_size=0.05
            )
        except AttributeError:
            # If add_point_cloud doesn't exist, just print the target positions
            print(f"Right hand target point at: {right_hand_target_point}")
            print(f"Left hand target point at: {left_hand_target_point}")
        
        try:
            # Calculate palm orientations to face target points
            right_hand_quaternion = calculate_palm_orientation_to_origin(ik_target_0.position, right_hand_target_point)
            left_hand_quaternion = calculate_palm_orientation_to_origin(ik_target_1.position, left_hand_target_point)
            
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([ik_target_0.position, ik_target_1.position]),
                target_wxyzs=np.array([right_hand_quaternion, left_hand_quaternion]),
            )
            
            # Create a new configuration that only updates upper body joints
            # Keep lower body joints at their initial values
            current_config = initial_config_array.copy()
            current_config[upper_body_indices] = solution[upper_body_indices]
            
            # Apply gripper states to hand joints
            joint_limits = urdf_vis.get_actuated_joint_limits()
            for i, joint_name in enumerate(actuated_joint_names):
                if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']:
                    # Get actual joint limits
                    lower, upper = joint_limits[joint_name]
                    lower = lower if lower is not None else -np.pi
                    upper = upper if upper is not None else np.pi
                    # Reverse the entire right hand direction
                    reversed_value = 1.0 - right_gripper_state
                    # For thumb joints (zero, one, two), reverse the direction again (double reverse = normal)
                    if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint']:
                        # Use normal value for thumb joints since we already reversed the whole hand
                        current_config[i] = lower + right_gripper_state * (upper - lower)
                    else:
                        # Use reversed value for other finger joints
                        current_config[i] = lower + reversed_value * (upper - lower)
                
                elif joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint']:
                    # Get actual joint limits
                    lower, upper = joint_limits[joint_name]
                    lower = lower if lower is not None else -np.pi
                    upper = upper if upper is not None else np.pi
                    # For thumb joints (zero, one, two), reverse the direction
                    if joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint']:
                        # Reverse the slider value for thumb joints
                        reversed_value = 1.0 - left_gripper_state
                        current_config[i] = lower + reversed_value * (upper - lower)
                    else:
                        # Map slider value (0-1) to joint position within actual limits
                        current_config[i] = lower + left_gripper_state * (upper - lower)
            
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