# python3 ik_vuer.py --path .venv/lib/python3.11/site-packages/mani_skill/assets/robots/g1_humanoid/g1.urdf  --port 8080 --flask-port 5000
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

def extract_hand_position(hand_data, hand_name):
    """
    Extract hand position from the hand tracking data.
    hand_data: Float32Array of 25 * 16 values (25 joints * 16 matrix values)
    hand_name: 'left' or 'right' for logging
    
    WebGL matrices are stored in column-major order:
    [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]
    
    Matrix layout:
    ⌈ a0 a4 a8  a12 ⌉
    | a1 a5 a9  a13 |
    | a2 a6 a10 a14 |
    ⌊ a3 a7 a11 a15 ⌋
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
    
    # print(f"{hand_name.capitalize()} hand position: {position}")
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

def apply_orientation_adjustment(position, roll_adjustment, pitch_adjustment, yaw_adjustment, hand_name="hand"):
    """
    Apply orientation adjustments to hand position to correct excessive angles.
    
    Args:
        position: Current hand position [x, y, z]
        roll_adjustment, pitch_adjustment, yaw_adjustment: Adjustment amounts
        hand_name: Name of the hand for logging
    
    Returns:
        adjusted_position: New position with orientation corrections applied
    """
    if position is None:
        return position
    
    # Convert adjustments to radians
    roll_rad = np.radians(roll_adjustment)
    pitch_rad = np.radians(pitch_adjustment)
    yaw_rad = np.radians(yaw_adjustment)
    
    # Apply small position adjustments based on orientation corrections
    # Roll affects Y and Z
    # Pitch affects X and Z  
    # Yaw affects X and Y
    adjusted_position = position.copy()
    
    # Apply roll adjustment (rotation around X-axis)
    if abs(roll_adjustment) > 0.001:
        adjusted_position[1] += roll_rad * 0.1  # Small Y adjustment
        adjusted_position[2] += roll_rad * 0.1  # Small Z adjustment
    
    # Apply pitch adjustment (rotation around Y-axis)
    if abs(pitch_adjustment) > 0.001:
        # adjusted_position[0] += pitch_rad * 0.1  # Small X adjustment
        adjusted_position[2] += pitch_rad * 0.1  # Small Z adjustment
    
    # Apply yaw adjustment (rotation around Z-axis)
    if abs(yaw_adjustment) > 0.001:
        adjusted_position[0] += yaw_rad * 0.1  # Small X adjustment
        adjusted_position[1] += yaw_rad * 0.1  # Small Y adjustment
    
    # print(f"{hand_name} position adjusted: {position} -> {adjusted_position}")
    return adjusted_position

def update_palm_orientation(current_roll, current_pitch, current_yaw, roll_adjustment, pitch_adjustment, yaw_adjustment, hand_name="hand"):
    """
    Update palm orientation based on adjustments.
    
    Args:
        current_roll, current_pitch, current_yaw: Current orientation angles in degrees
        roll_adjustment, pitch_adjustment, yaw_adjustment: Adjustment amounts in degrees
        hand_name: Name of the hand for logging
    
    Returns:
        (new_roll, new_pitch, new_yaw): Updated orientation angles
    """
    new_roll = current_roll + roll_adjustment
    new_pitch = current_pitch + pitch_adjustment
    new_yaw = current_yaw + yaw_adjustment
    
    # Keep angles within reasonable bounds (-180 to 180 degrees)
    new_roll = ((new_roll + 180) % 360) - 180
    new_pitch = ((new_pitch + 180) % 360) - 180
    new_yaw = ((new_yaw + 180) % 360) - 180
    
    if abs(roll_adjustment) > 0.001 or abs(pitch_adjustment) > 0.001 or abs(yaw_adjustment) > 0.001:
        print(f"{hand_name} orientation updated: roll={current_roll:.1f}° -> {new_roll:.1f}°, pitch={current_pitch:.1f}° -> {new_pitch:.1f}°, yaw={current_yaw:.1f}° -> {new_yaw:.1f}°")
    
    return new_roll, new_pitch, new_yaw

def detect_and_adjust_excessive_orientation(roll_deg, pitch_deg, yaw_deg, hand_name="hand", threshold=30.0, adjustment_amount=2.0):
    """
    Detect if hand orientation exceeds threshold and return constant adjustment amounts.
    
    Args:
        roll_deg, pitch_deg, yaw_deg: Current orientation angles in degrees
        hand_name: Name of the hand for logging
        threshold: Threshold angle in degrees (default 45.0)
        adjustment_amount: Constant adjustment amount in degrees (default 2.0)
    
    Returns:
        (roll_adjustment, pitch_adjustment, yaw_adjustment): Adjustment amounts
    """
    roll_adjustment = 0.0
    pitch_adjustment = 0.0
    yaw_adjustment = 0.0
    
    # Check if any angle exceeds the threshold
    if abs(roll_deg) > threshold:
        roll_adjustment = -np.sign(roll_deg) * adjustment_amount  # Move towards zero
        print(f"{hand_name} roll exceeded {threshold}° ({roll_deg:.1f}°), applying adjustment: {roll_adjustment:.1f}°")
    
    if abs(pitch_deg) > threshold:
        pitch_adjustment = -np.sign(pitch_deg) * adjustment_amount  # Move towards zero
        print(f"{hand_name} pitch exceeded {threshold}° ({pitch_deg:.1f}°), applying adjustment: {pitch_adjustment:.1f}°")
    
    if abs(yaw_deg) > threshold:
        yaw_adjustment = -np.sign(yaw_deg) * adjustment_amount  # Move towards zero
        print(f"{hand_name} yaw exceeded {threshold}° ({yaw_deg:.1f}°), applying adjustment: {yaw_adjustment:.1f}°")
    
    return roll_adjustment, pitch_adjustment, yaw_adjustment

def update_hand_joints_based_on_pinch(pinch_state, hand_joint_positions, joint_limits, hand_name, target_positions=None, lerp_factor=0.02):
    """
    Update hand joint positions based on pinch state with gradual interpolation.
    Uses the same logic as ik_keyboard.py for correct hand joint control.
    """
    global actuated_joint_names
    
    if pinch_state is None or hand_joint_positions is None:
        return hand_joint_positions
    
    # Convert pinch state to gripper value (0 = open, 1 = closed)
    gripper_value = pinch_state if pinch_state > 0.5 else 0.0
    
    # Calculate target positions based on pinch state
    target_positions = target_positions if target_positions is not None else hand_joint_positions.copy()
    
    # Define the hand joints for this specific hand
    if hand_name == 'left':
        hand_joint_names = ['left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint']
    else:  # right
        hand_joint_names = ['right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']
    
    # Calculate target positions for each joint with limited motion ranges
    for i, joint_name in enumerate(hand_joint_names):
        if i < len(target_positions):
            # Get joint limits for this specific joint
            joint_index = actuated_joint_names.index(joint_name) if joint_name in actuated_joint_names else -1
            if joint_index >= 0 and joint_index < len(joint_limits):
                lower, upper = joint_limits[joint_index]
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
            else:
                lower, upper = -np.pi, np.pi
            
            # Apply motion limits - restrict to half the joint range for more controlled movement
            motion_range = upper - lower
            limited_lower = lower + motion_range * 0.25  # Use 25% from the bottom
            limited_upper = upper - motion_range * 0.25  # Use 25% from the top (total 50% range)
            
            # Calculate target position based on hand and joint type with limited motion and reversed direction
            if hand_name == 'left':
                # For thumb joints (zero, one, two), use normal direction (reversed from previous logic)
                if joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint']:
                    target_positions[i] = limited_lower + gripper_value * (limited_upper - limited_lower)
                    if target_positions[i] < 0.0:
                        target_positions[i] = 0.0
                else:
                    # For other joints, reverse the direction
                    reversed_value = 1.0 - gripper_value
                    target_positions[i] = limited_lower + reversed_value * (limited_upper - limited_lower)
                    if target_positions[i] > 0.0:
                        target_positions[i] = 0.0
            
            else:  # right hand - reverse pinch direction
                # For thumb joints (zero, one, two), use reversed direction
                if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint']:
                    reversed_value = 1.0 - gripper_value
                    target_positions[i] = limited_lower + reversed_value * (limited_upper - limited_lower)
                    if target_positions[i] > 0.0:
                        target_positions[i] = 0.0
                else:
                    # For other joints, use normal direction
                    target_positions[i] = limited_lower + gripper_value * (limited_upper - limited_lower)
                    if target_positions[i] < 0.0:
                        target_positions[i] = 0.0
            
    
    # Gradually interpolate current positions towards target positions with much slower speed
    updated_positions = hand_joint_positions.copy()
    for i in range(len(updated_positions)):
        if i < len(target_positions):
            # Linear interpolation between current and target position with much slower lerp factor
            updated_positions[i] = hand_joint_positions[i] + lerp_factor * (target_positions[i] - hand_joint_positions[i])
    
    # print(f"{hand_name.capitalize()} hand pinch: {pinch_state:.2f}, gripper_value: {gripper_value:.2f}")
    return updated_positions, target_positions

@vuer_app.add_handler("HAND_MOVE")
async def handler(event, session):
    """Handle hand movement events and extract hand positions"""
    # print(f"Movement Event: key-{event.value.keys()}")

    hand_data = event.value
    
    if hand_data is not None:
        global initial_hand_tracking_left_position, initial_hand_tracking_right_position, initial_hand_tracking_left_orientation, initial_hand_tracking_right_orientation, left_hand_pinch_state, right_hand_pinch_state, left_hand_joint_positions, right_hand_joint_positions
        
        # Extract pinch states from hand tracking data
        try:
            if 'leftState' in hand_data.keys():
                left_hand_pinch_state = hand_data['leftState']['squeeze']
                # print(f"Left hand pinch state: {left_hand_pinch_state}")
                
                # Update left hand joints based on pinch state
                if left_hand_joint_positions is not None:
                    # print(f"Updating left hand joints with pinch state: {left_hand_pinch_state}")
                    global left_hand_target_positions
                    left_hand_joint_positions, left_hand_target_positions = update_hand_joints_based_on_pinch(
                        left_hand_pinch_state, left_hand_joint_positions, left_hand_joint_limits, 'left',
                        left_hand_target_positions if 'left_hand_target_positions' in globals() else None
                    )
                    # Store target positions for gradual updates
                    left_hand_target_positions = left_hand_target_positions
                    # print(f"Updated left hand joint positions: {left_hand_joint_positions}")
        except Exception as e:
            pass
        
        try:
            if 'rightState' in hand_data.keys():
                right_hand_pinch_state = hand_data['rightState']['squeeze']
                # print(f"Right hand pinch state: {right_hand_pinch_state}")
             
                # Update right hand joints based on pinch state
                if right_hand_joint_positions is not None:
                    # print(f"Updating right hand joints with pinch state: {right_hand_pinch_state}")
                    global right_hand_target_positions
                    right_hand_joint_positions, right_hand_target_positions = update_hand_joints_based_on_pinch(
                        right_hand_pinch_state, right_hand_joint_positions, right_hand_joint_limits, 'right',
                        right_hand_target_positions if 'right_hand_target_positions' in globals() else None
                    )
                    # Store target positions for gradual updates
                    right_hand_target_positions = right_hand_target_positions
                    # print(f"Updated right hand joint positions: {right_hand_joint_positions}")
        except Exception as e:
            pass

        if len(hand_data['left']) > 2:
            try:
                left_value = hand_data['left']
                left_position = extract_hand_position(left_value, 'left')
                
                # Extract left hand orientation
                left_roll, left_pitch, left_yaw = extract_hand_orientation(left_value, 'left')
                
                # Capture initial left hand position and orientation if not already captured
                if initial_hand_tracking_left_position is None and left_position is not None:
                    initial_hand_tracking_left_position = left_position.copy()
                    # print(f"Initial left hand position captured: {initial_hand_tracking_left_position}")
                
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
                
                # Check for excessive orientation and apply adjustments
                if left_roll is not None:
                    left_roll_adj, left_pitch_adj, left_yaw_adj = detect_and_adjust_excessive_orientation(
                        left_roll, left_pitch, left_yaw, "Left hand", 30.0, 2.0
                    )
                    
                    # Update palm orientation if adjustments are needed
                    if abs(left_roll_adj) > 0.001 or abs(left_pitch_adj) > 0.001 or abs(left_yaw_adj) > 0.001:
                        global current_left_palm_orientation
                        if current_left_palm_orientation is None:
                            current_left_palm_orientation = np.array([left_roll, left_pitch, left_yaw])
                        
                        new_roll, new_pitch, new_yaw = update_palm_orientation(
                            current_left_palm_orientation[0], current_left_palm_orientation[1], current_left_palm_orientation[2],
                            left_roll_adj, left_pitch_adj, left_yaw_adj, "Left hand"
                        )
                        current_left_palm_orientation = np.array([new_roll, new_pitch, new_yaw])
                
                # Calculate difference from initial position
                if initial_hand_tracking_left_position is not None and left_position is not None:
                    left_difference = left_position - initial_hand_tracking_left_position
                    # print(f"Left hand difference from initial: {left_difference}")
                    
                    # Update IK target 1 (left hand) based on difference from initial hand position
                    global ik_target_1_position, robot_initial_left_hand_position
                    if ik_target_1_position is not None and robot_initial_left_hand_position is not None:
                        # Use the difference from initial hand position as offset from robot's initial hand position
                        scaled_difference = left_difference * 0.5  # Scale factor for sensitivity
                        scaled_difference = np.array([-scaled_difference[2], -scaled_difference[0], scaled_difference[1]])
                        
                        # Use original XYZ coordinates without swapping
                        new_target_position = robot_initial_left_hand_position + scaled_difference
                        
                        # Apply orientation adjustments if needed
                        if left_roll is not None and (abs(left_roll_adj) > 0.001 or abs(left_pitch_adj) > 0.001 or abs(left_yaw_adj) > 0.001):
                            new_target_position = apply_orientation_adjustment(
                                new_target_position, left_roll_adj, left_pitch_adj, left_yaw_adj, "Left hand"
                            )
                        
                        ik_target_1_position = new_target_position.copy()  # Update global target
                        # print(f"Updated left IK target: initial={robot_initial_left_hand_position}, diff={scaled_difference}, new={new_target_position}")
                    
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
                    # print(f"Initial right hand position captured: {initial_hand_tracking_right_position}")
                
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
                
                # Check for excessive orientation and apply adjustments
                if right_roll is not None:
                    right_roll_adj, right_pitch_adj, right_yaw_adj = detect_and_adjust_excessive_orientation(
                        right_roll, right_pitch, right_yaw, "Right hand", 30.0, 2.0
                    )
                    
                    # Update palm orientation if adjustments are needed
                    if abs(right_roll_adj) > 0.001 or abs(right_pitch_adj) > 0.001 or abs(right_yaw_adj) > 0.001:
                        global current_right_palm_orientation
                        if current_right_palm_orientation is None:
                            current_right_palm_orientation = np.array([right_roll, right_pitch, right_yaw])
                        
                        new_roll, new_pitch, new_yaw = update_palm_orientation(
                            current_right_palm_orientation[0], current_right_palm_orientation[1], current_right_palm_orientation[2],
                            right_roll_adj, right_pitch_adj, right_yaw_adj, "Right hand"
                        )
                        current_right_palm_orientation = np.array([new_roll, new_pitch, new_yaw])
                
                # Calculate difference from initial position
                if initial_hand_tracking_right_position is not None and right_position is not None:
                    right_difference = right_position - initial_hand_tracking_right_position
                    # print(f"Right hand difference from initial: {right_difference}")
                    
                    # Update IK target 0 (right hand) based on difference from initial hand position
                    global ik_target_0_position, robot_initial_right_hand_position
                    if ik_target_0_position is not None and robot_initial_right_hand_position is not None:
                        # Use the difference from initial hand position as offset from robot's initial hand position
                        scaled_difference = right_difference * 0.5  # Scale factor for sensitivity
                        scaled_difference = np.array([-scaled_difference[2], -scaled_difference[0], scaled_difference[1]])
                        
                        # Use original XYZ coordinates without swapping
                        new_target_position = robot_initial_right_hand_position + scaled_difference
                        
                        # Apply orientation adjustments if needed
                        if right_roll is not None and (abs(right_roll_adj) > 0.001 or abs(right_pitch_adj) > 0.001 or abs(right_yaw_adj) > 0.001):
                            new_target_position = apply_orientation_adjustment(
                                new_target_position, right_roll_adj, right_pitch_adj, right_yaw_adj, "Right hand"
                            )
                        
                        ik_target_0_position = new_target_position.copy()  # Update global target
                        # print(f"Updated right IK target: initial={robot_initial_right_hand_position}, diff={scaled_difference}, new={new_target_position}")
                    
            except Exception as e:
                print(f"Error accessing 'right': {e}")
        
    
    # # Print hand states if available
    # if hasattr(hand_data, 'leftState'):
    #     print(f"Left hand state: {hand_data.leftState}")
    
    # if hasattr(hand_data, 'rightState'):
    #     print(f"Right hand state: {hand_data.rightState}")
    
    # print("-" * 50)

@vuer_app.add_handler("test")
async def test_handler(event, session):
    """Test handler to verify Vuer is working"""
    print(f"Test event received: {event.key} - {event.value}")

@vuer_app.add_handler("*")
async def debug_all_events(event, session):
    """Debug handler to see all events"""
    print(f"DEBUG EVENT: {event.key} - {type(event.value)} - {event.value}")




# Global variables to store robot state
current_robot_config = None
actuated_joint_names = None

# Global variables to store initial hand positions from hand tracking
initial_hand_tracking_left_position = None
initial_hand_tracking_right_position = None

# Global variables to store initial hand orientations from hand tracking
initial_hand_tracking_left_orientation = None
initial_hand_tracking_right_orientation = None

# Global variables to store current palm orientations for IK targets
current_left_palm_orientation = None
current_right_palm_orientation = None

# Global variables to store IK targets for hand tracking control
ik_target_0_position = None
ik_target_1_position = None

# Global variables to store robot's initial hand positions (from default joint configuration)
robot_initial_right_hand_position = None
robot_initial_left_hand_position = None

# Global variables to store pinch states from hand tracking
left_hand_pinch_state = None
right_hand_pinch_state = None

# Global variables to store current hand joint positions for gradual closing
left_hand_joint_positions = None
right_hand_joint_positions = None

# Global variables to store hand joint limits
left_hand_joint_limits = None
right_hand_joint_limits = None

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

@app.route('/vuer_status', methods=['GET'])
def vuer_status():
    """Check if Vuer server is running"""
    return jsonify({
        "vuer_running": True,
        "handlers_registered": ["HAND_MOVE", "test"],
        "timestamp": time.time()
    })

@app.route('/initial_hand_tracking_positions', methods=['GET'])
def get_initial_hand_tracking_positions():
    """Return the initial hand tracking positions as JSON"""
    global initial_hand_tracking_left_position, initial_hand_tracking_right_position
    
    return jsonify({
        "left_hand": initial_hand_tracking_left_position.tolist() if initial_hand_tracking_left_position is not None else None,
        "right_hand": initial_hand_tracking_right_position.tolist() if initial_hand_tracking_right_position is not None else None,
        "timestamp": time.time()
    })

@app.route('/initial_hand_tracking_orientations', methods=['GET'])
def get_initial_hand_tracking_orientations():
    """Return the initial hand tracking orientations as JSON"""
    global initial_hand_tracking_left_orientation, initial_hand_tracking_right_orientation
    
    return jsonify({
        "left_hand": initial_hand_tracking_left_orientation.tolist() if initial_hand_tracking_left_orientation is not None else None,
        "right_hand": initial_hand_tracking_right_orientation.tolist() if initial_hand_tracking_right_orientation is not None else None,
        "timestamp": time.time()
    })

@app.route('/hand_pinch_states', methods=['GET'])
def get_hand_pinch_states():
    """Return the current hand pinch states as JSON"""
    global left_hand_pinch_state, right_hand_pinch_state
    
    return jsonify({
        "left_hand_pinch": left_hand_pinch_state,
        "right_hand_pinch": right_hand_pinch_state,
        "timestamp": time.time()
    })

def run_flask_server(flask_port=5000):
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

def main(path: str, port: int, flask_port: int = 5000) -> None:
    global current_robot_config, actuated_joint_names, ik_target_0_position, ik_target_1_position, left_hand_joint_positions, right_hand_joint_positions, left_hand_joint_limits, right_hand_joint_limits
    
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
    # print(f"Initial configuration set with {len(initial_config)} joints")
    
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
    
    # Initialize global IK target positions for hand tracking control
    global ik_target_0_position, ik_target_1_position, robot_initial_right_hand_position, robot_initial_left_hand_position
    ik_target_0_position = right_palm_pos.copy()
    ik_target_1_position = left_palm_pos.copy()
    robot_initial_right_hand_position = right_palm_pos.copy()
    robot_initial_left_hand_position = left_palm_pos.copy()
    # print(f"Initialized IK target positions:")
    # print(f"  Right target: {ik_target_0_position}")
    # print(f"  Left target: {ik_target_1_position}")
    # print(f"Robot initial hand positions:")
    # print(f"  Right hand: {robot_initial_right_hand_position}")
    # print(f"  Left hand: {robot_initial_left_hand_position}")
    
    # Initialize hand joint positions and limits
    global left_hand_joint_positions, right_hand_joint_positions, left_hand_joint_limits, right_hand_joint_limits
    
    # Get joint limits for hand joints (assuming last few joints are hand joints)
    joint_limits = list(urdf_vis.get_actuated_joint_limits().values())
    
    # Print all joint names to identify hand joints
    # print(f"All actuated joint names: {actuated_joint_names}")
    # print(f"Total joints: {len(actuated_joint_names)}")
    
    # Find hand joints by looking for "left_zero_joint" to "left_six_joint" and "right_zero_joint" to "right_six_joint"
    left_hand_joint_indices = []
    right_hand_joint_indices = []
    
    for i, joint_name in enumerate(actuated_joint_names):
        if joint_name.startswith('left_') and joint_name.endswith('_joint'):
            left_hand_joint_indices.append(i)
            # print(f"Found left hand joint: {joint_name} at index {i}")
        elif joint_name.startswith('right_') and joint_name.endswith('_joint'):
            right_hand_joint_indices.append(i)
            # print(f"Found right hand joint: {joint_name} at index {i}")
    
    # Initialize hand joint positions and limits
    if len(left_hand_joint_indices) > 0 and len(right_hand_joint_indices) > 0:
        # print(f"Using {len(left_hand_joint_indices)} left hand joints and {len(right_hand_joint_indices)} right hand joints")
        
        # Get left hand joint positions and limits
        left_hand_joint_positions = initial_config_array[left_hand_joint_indices].copy()
        left_hand_joint_limits = [joint_limits[i] for i in left_hand_joint_indices]
        
        # Get right hand joint positions and limits
        right_hand_joint_positions = initial_config_array[right_hand_joint_indices].copy()
        right_hand_joint_limits = [joint_limits[i] for i in right_hand_joint_indices]
    else:
        # Fallback: assume last 6 joints are hand joints
        # print("No hand joints found by name, using last 6 joints as fallback")
        hand_joint_count = 6  # 3 joints per hand
        left_hand_joint_positions = initial_config_array[-hand_joint_count:-hand_joint_count//2].copy()
        right_hand_joint_positions = initial_config_array[-hand_joint_count//2:].copy()
        left_hand_joint_limits = joint_limits[-hand_joint_count:-hand_joint_count//2]
        right_hand_joint_limits = joint_limits[-hand_joint_count//2:]
    
    print(f"Initialized hand joint positions:")
    print(f"  Left hand joints: {left_hand_joint_positions}")
    print(f"  Right hand joints: {right_hand_joint_positions}")
    print(f"  Left hand limits: {left_hand_joint_limits}")
    print(f"  Right hand limits: {right_hand_joint_limits}")
    
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
    
    # Start Vuer hand tracking
    print("Starting Vuer hand tracking...")
    print("Make sure you have SSL certificates set up for hand tracking to work!")
    print("You can access the hand tracking interface at: https://localhost:8080")
    
    # Start Vuer app
    # vuer_app.run()
    
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
    
    # Start Vuer app in a separate thread with proper event loop
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
        
        # Use hand tracking controlled targets if available, otherwise use transform controls
        right_target_pos = ik_target_0_position if ik_target_0_position is not None else ik_target_0.position
        left_target_pos = ik_target_1_position if ik_target_1_position is not None else ik_target_1.position
        
        # Update the visual transform controls to match hand tracking positions
        if ik_target_0_position is not None:
            ik_target_0.position = right_target_pos
        if ik_target_1_position is not None:
            ik_target_1.position = left_target_pos
        
        try:
            # Use updated palm orientations if available, otherwise use default
            right_wxyz = ik_target_0.wxyz
            left_wxyz = ik_target_1.wxyz
            
            # Convert palm orientations to quaternions if available
            if current_right_palm_orientation is not None:
                # Convert euler angles to quaternion (simplified conversion)
                right_roll_rad = np.radians(current_right_palm_orientation[0])
                right_pitch_rad = np.radians(current_right_palm_orientation[1])
                right_yaw_rad = np.radians(current_right_palm_orientation[2])
                
                # Simple quaternion from euler angles (not perfect but functional)
                cy = np.cos(right_yaw_rad * 0.5)
                sy = np.sin(right_yaw_rad * 0.5)
                cp = np.cos(right_pitch_rad * 0.5)
                sp = np.sin(right_pitch_rad * 0.5)
                cr = np.cos(right_roll_rad * 0.5)
                sr = np.sin(right_roll_rad * 0.5)
                
                right_wxyz = np.array([
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                ])
            
            if current_left_palm_orientation is not None:
                # Convert euler angles to quaternion (simplified conversion)
                left_roll_rad = np.radians(current_left_palm_orientation[0])
                left_pitch_rad = np.radians(current_left_palm_orientation[1])
                left_yaw_rad = np.radians(current_left_palm_orientation[2])
                
                # Simple quaternion from euler angles (not perfect but functional)
                cy = np.cos(left_yaw_rad * 0.5)
                sy = np.sin(left_yaw_rad * 0.5)
                cp = np.cos(left_pitch_rad * 0.5)
                sp = np.sin(left_pitch_rad * 0.5)
                cr = np.cos(left_roll_rad * 0.5)
                sr = np.sin(left_roll_rad * 0.5)
                
                left_wxyz = np.array([
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                ])
            
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([right_target_pos, left_target_pos]),
                target_wxyzs=np.array([right_wxyz, left_wxyz]),
            )
            
            # Create a new configuration that only updates upper body joints
            # Keep lower body joints at their initial values
            current_config = initial_config_array.copy()
            current_config[upper_body_indices] = solution[upper_body_indices]
            
            # Apply hand joint positions based on pinch states
            if left_hand_joint_positions is not None:
                # print(f"Applying left hand joints: {left_hand_joint_positions}")
                # Find left hand joint indices
                left_hand_joint_indices = []
                for i, joint_name in enumerate(actuated_joint_names):
                    if joint_name in ['left_zero_joint', 'left_one_joint', 'left_two_joint', 'left_three_joint', 'left_four_joint', 'left_five_joint', 'left_six_joint']:
                        left_hand_joint_indices.append(i)
                
                if len(left_hand_joint_indices) > 0:
                    # Apply left hand joint positions
                    step = 0
                    for i, pos in zip(left_hand_joint_indices, left_hand_joint_positions):
                        if step == 0:
                            step += 1
                        else:
                            current_config[i] = pos
                        # print(f"Set left hand joint {i} ({actuated_joint_names[i]}) to {pos}")
            
            if right_hand_joint_positions is not None:
                # print(f"Applying right hand joints: {right_hand_joint_positions}")
                # Find right hand joint indices
                right_hand_joint_indices = []
                for i, joint_name in enumerate(actuated_joint_names):
                    if joint_name in ['right_zero_joint', 'right_one_joint', 'right_two_joint', 'right_three_joint', 'right_four_joint', 'right_five_joint', 'right_six_joint']:
                        right_hand_joint_indices.append(i)
                
                if len(right_hand_joint_indices) > 0:
                    # Apply right hand joint positions
                    step = 0
                    for i, pos in zip(right_hand_joint_indices, right_hand_joint_positions):
                        if step == 0:
                            step += 1
                        else:
                            current_config[i] = pos
                        # print(f"Set right hand joint {i} ({actuated_joint_names[i]}) to {pos}")
            
            # Store current configuration globally for Flask API
            current_robot_config = current_config.copy()
            
            urdf_vis.update_cfg(current_config)
            
        except Exception as e:
            print(f"IK solver failed: {e}")
            import traceback
            traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)