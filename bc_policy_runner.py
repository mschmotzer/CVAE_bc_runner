#!/usr/bin/env python3
"""
Behavior Cloning Policy Runner for Franka Robot
This node loads a trained BC policy using robomimic and runs inference on the Franka robot.
"""
from html import parser
import os
import pickle
import numpy as np
import torch
import argparse
import rclpy
import threading
import signal
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
# Import Functionality from utils folder
from utils.gripper_control import GripperControlMixin
from utils.cube_manager import CubeManagerMixin
from utils.observations import ObservationMixin
from utils.keyboard import KeyboardInput
from utils.keyboard_handler import KeyboardHandlerMixin
from utils.instructions import InstructionMixin
from utils.policy_control import PolicyControlMixin
from utils.policy_control_cvae  import PolicyControlMixin_cvae
from utils.robot_state import RobotStateMixin
import sys
import pyzed.sl as sl
sys.path.append('/home/pdz/MasterThesis_MSC/Isaaclab_5/IsaacLab/')

# BCPolicy Runner Nodex
class BCPolicyRunner(Node, GripperControlMixin, CubeManagerMixin, ObservationMixin, InstructionMixin, KeyboardHandlerMixin, PolicyControlMixin, RobotStateMixin):
    """ROS2 Node for running Behavior Cloning policy on Franka robot"""
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True, 
                 control_frequency: float = 20.0, testing_mode: bool = False, context_length: int = 1, camera_enabled: bool = False, args_cli=None):
        self.context_length = context_length
        self.velocity_control = args_cli.velocity_control
        super().__init__('bc_policy_runner')
        
        # Store testing mode
        self.testing_mode = testing_mode
        
        # Initialize parameters
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_frequency = control_frequency
        self.step_count = 0
        self.camera_enabled = camera_enabled
        self.cvae = args_cli.cvae   
  
        # Add trial tracking
        self.trial_id = 0
        self.current_config_name = "Default"
        self.num_cameras = args_cli.num_cameras if args_cli.cvae else 0
        # Initialize keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Load policy using robomimic framework and set it to evaluation mode
        if args_cli.cvae:
            self.policy = cvae_policy_loader(policy_path, device=device, args_cli=args_cli)
        else:
            self.policy, self.ckpt_dict = self.load_policy(policy_path, device=device)


        # Initialize for new episode
        if not self.cvae:   
            self.policy.start_episode()  
        else:
            self.policy.cuda()
            self.policy.eval()

        # Robot state storage
        self.current_eef_pose = None
        self.current_gripper_positions = None
        self.current_gripper_velocities = None
        self.current_jacobian = None
        self.joint_pos = None
        self.joint_vel = None
        self.img1 = None
        self.img2 = None
        # Object state storage
        self.cube_positions = {
            'cube_1': np.array([0.400, -0.200, 0.0203]),
            'cube_2': np.array([0.475, -0.046, 0.0203]),
            'cube_3': np.array([0.430, -0.279, 0.0203])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format "IsaacLab expects quaternions in [w, x, y, z] format"
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])   # Identity quaternion
        }


        # Dynamic object state storage
        self.cube_attached = None # None, 'cube_2', 'cube_3'
        self.last_gripper_state = 'open' # Track gripper state changes
        self.grasp_threshold = 0.06 # Width below which we consider the gripper "closed"
        self.release_threshold = 0.07 # Width above which we consider gripper "open"
        self.proximity_threshold = 0.05 # Distance threshold for grasp condition 
        self.grasp_sequence_count = 0 # 0: no grasps, 1: first grasp (cube_2) 2: second grasp (cube_3)
        self.grasp_min = 0.0475
        self.grasp_max = 0.0495
        self.ee_pos_cube_vicinity = np.array([0.0, 0.0, 0.0])  # Offset from EE to cube when attached
        self.gripper_inrange = 4
        self.counter = 0
        # Add cube spawning configuration
        self.cube_spawn_config = {
            "pose_range": {
                "x": (0.4, 0.6),
                "y": (-0.3, 0.3), 
                "z": (0.0203, 0.0203)
            },
            "min_cube_distance": 0.05  # Minimum distance between cubes to avoid overlap
        }

        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])  # Franka's home position in the world frame

        # Control flags
        self.is_running = False
        self.episode_active = False
        self.policy_running = False
        self.shutdown_requested = False

        # Camera Flags
        self.camera_cube_poses = {}
        self.camera_cube_poses_received = False
        
        # Keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.07 # Max width for Franka Hand
        self.gripper_speed = 0.6 # Default speed (m/s)
        self.gripper_force = 40.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05
        self.gripper_epsilon_outer = 0.06
        self.initialize_gripper_clients()
        self.gripper_action_in_progress = False
        self.gripper_action_lock = threading.Lock()
        self.gripper_last_command_time = 0.0
        self.gripper_command_cooldown = 1.0  # 1 second between commands
        self.cube_ee_offset = np.array([0.0, 0.0, 0.0])  # Offset from EE to cube when attached
        self.obs_dict= None
        
        if self.cvae:
            self.stats_path = os.path.join(args_cli.data_norm, "dataset_stats.pkl")
            with open(self.stats_path, "rb") as f:
                stats = pickle.load(f)

            qpos_mean = stats["qpos_mean"]
            qpos_std = stats["qpos_std"]
            if self.velocity_control:
                qvel_mean = stats["qvel_mean"]
                qvel_std = stats["qvel_std"]
            action_mean = stats["action_mean"]
            action_std = stats["action_std"]

            self.pre_process_qpos = lambda q: (q - qpos_mean) / qpos_std
            self.pre_process_qvel = lambda q: (q - qvel_mean) / qvel_std if self.velocity_control else None
            self.post_process = lambda a: a * action_std + action_mean
        # Setup QoS (Quality of Service) profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # ------------------------------------------------------Subscribers--------------------------------------------------------
        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.eef_pose_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )



        
        # Subscribe to cube poses from camera
        self.setup_camera_subscribers(qos_profile, self.callback_group)
        topic_name = '/redcube_position'
        topics_and_types = self.get_topic_names_and_types()
        topic_exists = any(topic[0] == topic_name for topic in topics_and_types)

        if self.camera_enabled and topic_exists:
            self.redcube_sub = self.create_subscription(
                PoseStamped,
                '/redcube_position',
                self.redcube_callback,
                qos_profile,
                callback_group=self.callback_group
            )

            self.bluecube_sub = self.create_subscription(
                PoseStamped,
                '/bluecube_position',
                self.bluecube_callback,
                qos_profile,
                callback_group=self.callback_group
            )

            self.greencube_sub = self.create_subscription(
                PoseStamped,
                '/greencube_position',
                self.greencube_callback,
                qos_profile,
                callback_group=self.callback_group
            )
        elif self.cvae:
            """devices = sl.Camera.get_device_list()
            detected = len(devices)  
            if detected != self.num_cameras:
                self.get_logger().error(f"ZED Cameras detected: {detected}, but {self.num_cameras} required.")
                raise RuntimeError("Insufficient number of ZED cameras connected.")"""
            from cv_bridge import CvBridge
            
            self.bridge = CvBridge()
            self.img1_msg = self.create_subscription(
                Image,
                '/franka/jacobian_ee',
                self.image1_callback,
                qos_profile,
                callback_group=self.callback_group
            )
            self.img2_msg = self.create_subscription(
                Image,
                '/zed_cam2/image_raw',
                self.image2_callback,
                qos_profile,
                callback_group=self.callback_group
            )

        else:
            self.get_logger().warn(f"Camera cube position topics do not exist back to hardcoded cube positions")
            self.camera_enabled = False

        # ------------------------------------------------------Publishers---------------------------------------------------------------
        self.pose_command_pub = self.create_publisher(
            Float64MultiArray,
            '/cartesian_position_controller/commands',
            qos_profile
        )

        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_position_controller/commands',
            qos_profile
        )

        # --------------------------------- ROS2 Timer for the Control Loop ---------------------------------
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,  # Period = 1/20Hz = 0.05 seconds
            self.control_loop, 
            callback_group=self.callback_group,
            clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        )
        
        # --------------------------------- Keyboard Input Timer for Policy Testing and Normal Mode ---------
        if self.testing_mode and not self.camera_enabled:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input_testing)  # ADD THIS
        elif self.camera_enabled:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input_camera)
        else:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        # Print initial instructions based on mode
        if self.testing_mode:
            self.print_instructions_testing()
        elif self.camera_enabled:
            self.print_instructions_camera()
        else:
            self.print_instructions()

        
    # -------------------------------Main Control Loop Logic-------------------------------------------------
    def control_loop(self):
        """Main control loop for normal mode"""
        if self.shutdown_requested:
            return
        if self.is_running and self.episode_active:
            try:
                self.handle_control_step()
            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")

    def handle_control_step(self):
        """Handle a single step of the control loop"""
        try:
            # STEP 1: Check for gripper state changes and handle cube attachment
            if not self.camera_enabled:
                gripper_state_change = self.detect_gripper_state_change()
                if gripper_state_change == 'closed' and self.cube_attached is None:
                    self.handle_cube_attachment(gripper_state_change)
                    if self.cube_attached is not None and self.current_eef_pose is not None:
                        self.update_attached_cube_pose()
                elif gripper_state_change == 'open':
                    self.cube_attached = None
                    self.counter = 0
                    self.cube_ee_offset = np.array([0.0, 0.0, 0.0])
                elif self.cube_attached is not None and self.current_eef_pose is not None:
                    self.update_attached_cube_pose()
            
            # STEP 3: Calculate manipulability index
            manipulability_index = self.calculate_manipulability_index()
            # STEP 4: Create observation for the policy
            if not self.cvae:
                obs = self.create_observation(manipulability_index)
            else:
                obs = self.create_observation_cvae()
                obs.to(self.policy.device)
            if self.obs_dict is not None:

                # STEP 5: Run policy inference
                action = self.policy(obs)
                # Convert action to numpy array if needed
                action =self.post_process(action) if self.cvae else action
                action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
                # STEP 6: Save observation for analysis
                #self.save_observation_to_csv(obs, action_np, manipulability_index)
                
                # STEP 7: Log observation and action together
                #self.log_observation_compact(obs, action_np, manipulability_index)
                
                # STEP 8: Execute the action
                print("Exectuing action:", action_np)
                #self.execute_action(action_np)
                #print(f"frequency: {1.0/(time_end-time_start)}")
                # If gripper_state_change is not None, handle cube attachment logic

            

        except Exception as e:
            self.get_logger().error(f"Normal mode execution error: {e}")
class BCPolicyRunner_cvae(Node, GripperControlMixin, CubeManagerMixin, ObservationMixin, InstructionMixin, KeyboardHandlerMixin, PolicyControlMixin_cvae, RobotStateMixin):
    """ROS2 Node for running Behavior Cloning policy on Franka robot"""
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True, 
                 control_frequency: float = 20.0, testing_mode: bool = False, context_length: int = 1, camera_enabled: bool = False, args_cli=None):
        self.context_length = context_length
        self.velocity_control = args_cli.velocity_control
        super().__init__('bc_policy_runner')
        
        # Store testing mode
        self.testing_mode = testing_mode
        
        # Initialize parameters
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_frequency = control_frequency
        self.step_count = 0
        self.camera_enabled = camera_enabled
        self.cvae = args_cli.cvae   

        # Add trial tracking
        self.trial_id = 0
        self.current_config_name = "Default"
        self.num_cameras = args_cli.num_cameras if args_cli.cvae else 0
        # Initialize keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Load policy using robomimic framework and set it to evaluation mode
        if args_cli.cvae:
            self.policy = self.cvae_policy_loader(policy_path, args_cli=args_cli)
        else:
            self.policy, self.ckpt_dict = self.load_policy(policy_path, device=device)


        self.policy.cuda()
        self.policy.eval()

        # Robot state storage
        self.current_eef_pose = None
        self.current_gripper_positions = None
        self.current_gripper_velocities = None
        self.current_jacobian = None
        self.joint_pos = None
        self.joint_vel = None
        self.img1 = None
        self.img2 = None
        # Object state storage
        self.cube_positions = {
            'cube_1': np.array([0.400, -0.200, 0.0203]),
            'cube_2': np.array([0.475, -0.046, 0.0203]),
            'cube_3': np.array([0.430, -0.279, 0.0203])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format "IsaacLab expects quaternions in [w, x, y, z] format"
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])   # Identity quaternion
        }


        # Dynamic object state storage
        self.cube_attached = None # None, 'cube_2', 'cube_3'
        self.last_gripper_state = 'open' # Track gripper state changes
        self.grasp_threshold = 0.06 # Width below which we consider the gripper "closed"
        self.release_threshold = 0.07 # Width above which we consider gripper "open"
        self.proximity_threshold = 0.05 # Distance threshold for grasp condition 
        self.grasp_sequence_count = 0 # 0: no grasps, 1: first grasp (cube_2) 2: second grasp (cube_3)
        self.grasp_min = 0.0475
        self.grasp_max = 0.0495
        self.ee_pos_cube_vicinity = np.array([0.0, 0.0, 0.0])  # Offset from EE to cube when attached
        self.gripper_inrange = 4
        self.counter = 0
        # Add cube spawning configuration
        self.cube_spawn_config = {
            "pose_range": {
                "x": (0.4, 0.6),
                "y": (-0.3, 0.3), 
                "z": (0.0203, 0.0203)
            },
            "min_cube_distance": 0.05  # Minimum distance between cubes to avoid overlap
        }

        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])  # Franka's home position in the world frame

        # Control flags
        self.is_running = False
        self.episode_active = False
        self.policy_running = False
        self.shutdown_requested = False

        # Camera Flags
        self.camera_cube_poses = {}
        self.camera_cube_poses_received = False
        
        # Keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.07 # Max width for Franka Hand
        self.gripper_speed = 0.6 # Default speed (m/s)
        self.gripper_force = 40.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05
        self.gripper_epsilon_outer = 0.06
        self.initialize_gripper_clients()
        self.gripper_action_in_progress = False
        self.gripper_action_lock = threading.Lock()
        self.gripper_last_command_time = 0.0
        self.gripper_command_cooldown = 1.0  # 1 second between commands
        self.cube_ee_offset = np.array([0.0, 0.0, 0.0])  # Offset from EE to cube when attached
        self.obs_dict= None
        
        if self.cvae:
            self.stats_path = os.path.join(args_cli.data_norm, "dataset_stats.pkl")
            with open(self.stats_path, "rb") as f:
                stats = pickle.load(f)

            qpos_mean = stats["qpos_mean"]
            qpos_std = stats["qpos_std"]
            if self.velocity_control:
                qvel_mean = stats["qvel_mean"]
                qvel_std = stats["qvel_std"]
            action_mean = stats["action_mean"]
            action_std = stats["action_std"]

            self.pre_process_qpos = lambda q: (q - qpos_mean) / qpos_std
            self.pre_process_qvel = lambda q: (q - qvel_mean) / qvel_std if self.velocity_control else None
            self.post_process = lambda a: a * action_std + action_mean
        # Setup QoS (Quality of Service) profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # ------------------------------------------------------Subscribers--------------------------------------------------------
        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.eef_pose_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )

        
        # Subscribe to cube poses from camera
        self.setup_camera_subscribers(qos_profile, self.callback_group)
        topic_name = '/redcube_position'
        topics_and_types = self.get_topic_names_and_types()
        topic_exists = any(topic[0] == topic_name for topic in topics_and_types)


        if self.cvae:
            """devices = sl.Camera.get_device_list()
            detected = len(devices)  
            if detected != self.num_cameras:
                self.get_logger().error(f"ZED Cameras detected: {detected}, but {self.num_cameras} required.")
                raise RuntimeError("Insufficient number of ZED cameras connected.")"""
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.img1_msg = self.create_subscription(
                Image,
                '/zed_cam1/image_raw',
                self.image1_callback,
                qos_profile,
                callback_group=self.callback_group
            )
            self.img2_msg = self.create_subscription(
                Image,
                '/zed_cam2/image_raw',
                self.image2_callback,
                qos_profile,
                callback_group=self.callback_group
            )

        else:
            self.get_logger().warn(f"Camera  topics do not exist back to hardcoded cube positions")

        # ------------------------------------------------------Publishers---------------------------------------------------------------
        self.pose_command_pub = self.create_publisher(
            Float64MultiArray,
            '/cartesian_position_controller/commands',
            qos_profile
        )

        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_position_controller/commands',
            qos_profile
        )

        # --------------------------------- ROS2 Timer for the Control Loop ---------------------------------
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,  # Period = 1/20Hz = 0.05 seconds
            self.control_loop, 
            callback_group=self.callback_group,
            clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        )
        
        # --------------------------------- Keyboard Input Timer for Policy Testing and Normal Mode ---------
        if self.testing_mode and not self.camera_enabled:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input_testing)  # ADD THIS
        elif self.camera_enabled:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input_camera)
        else:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        # Print initial instructions based on mode
        if self.testing_mode:
            self.print_instructions_testing()
        elif self.camera_enabled:
            self.print_instructions_camera()
        else:
            self.print_instructions()

        
    # -------------------------------Main Control Loop Logic-------------------------------------------------
    def control_loop(self):
        """Main control loop for normal mode"""
        if self.shutdown_requested:
            return
        if self.is_running and self.episode_active:
            try:
                self.handle_control_step()
            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")

    def handle_control_step(self):
        """Handle a single step of the control loop"""
        try:
            # STEP 1: Check for gripper state changes and handle cube attachment
            if not self.camera_enabled:
                gripper_state_change = self.detect_gripper_state_change()
                if gripper_state_change == 'closed' and self.cube_attached is None:
                    self.handle_cube_attachment(gripper_state_change)
                    if self.cube_attached is not None and self.current_eef_pose is not None:
                        self.update_attached_cube_pose()
                elif gripper_state_change == 'open':
                    self.cube_attached = None
                    self.counter = 0
                    self.cube_ee_offset = np.array([0.0, 0.0, 0.0])
                elif self.cube_attached is not None and self.current_eef_pose is not None:
                    self.update_attached_cube_pose()
            
            # STEP 3: Calculate manipulability index
            manipulability_index = self.calculate_manipulability_index()
            # STEP 4: Create observation for the policy
            if not self.cvae:
                obs = self.create_observation(manipulability_index)
            else: 
                obs = self.create_observation_cvae()
                pos = obs['joint_pos']#.to(self.policy.device)
                vel = obs['joint_vel']#.to(self.policy.device) if self.velocity_control else None
                image = obs['image']#.to(self.policy.device) 
            if self.obs_dict is not None:
             
                action = self.policy(pos.unsqueeze(0).cuda(), image.unsqueeze(0).cuda(), qvel = vel.unsqueeze(0).cuda() if self.velocity_control else None)
                # Convert action to numpy array if needed
                k=0
                N = 32 #action.shape[1]
                weights = torch.exp(-k * torch.arange(N, device=torch.device(action.device)))
                print("Weights:", weights[0])
                print("Weighted actions:", weights.shape, action.shape)   
                weights = (weights / weights.sum()).unsqueeze(1)
                #gripper = post_process(actions_t)[:,-1].sum(dim=0)
                #gripper_action = -1 if gripper < 0 else 1
                raw_action = (action[0,:N,:] * weights).sum(dim=0, keepdim=True)  # weighted sum
                # Move to CPU and convert to NumPy for post-processing
                action_np = raw_action.detach().cpu().numpy()  # shape [1, action_dim]
                # Apply post-processing (expects NumPy)
                action_np = self.post_process(action_np)       # still NumPy
                # If your robot expects shape [1, 8] ensure it
                action_np = action_np.reshape(1, -1)          # [1, 8]
                # Execute
                print("Post-processed action:", action_np)
                self.execute_action(action_np)
                
                #print(f"frequency: {1.0/(time_end-time_start)}")
                # If gripper_state_change is not None, handle cube attachment logic

            

        except Exception as e:
            self.get_logger().error(f"Normal mode execution error: {e}")

    
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BC Policy Runner for Franka Robot")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pth)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy inference")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="Control frequency in Hz")
    parser.add_argument("--testing", action="store_true",
                       help="Run in testing mode with systematic configurations")
    parser.add_argument("--context_length", type=int, default=1,
                       help="Run with longer context")
    parser.add_argument("--camera", action="store_true",
                       help="Enable camera input for the policy")
    parser.add_argument('--velocity_control', action='store_true')
    parser.add_argument('--action_length', type=int, default=1, help='action_length')
    parser.add_argument('--cvae', action='store_true', help='Enable CVAE')
    parser.add_argument("--data_norm", type=str, default=None, required=False,
                       help="Path to normalize file")
    parser.add_argument(
    '--num_cameras',
    type=int,
    help='Number of cameras (required if --cvae is set)'
    )

    args = parser.parse_args()

    if args.cvae and (args.num_cameras is None or args.data_norm is None):
        parser.error('--num_cameras and/or --data_norm is required when --cvae is set')
        raise ValueError('--num_cameras and/or --data_norm is required when --cvae is set')
    if args.cvae and args.num_cameras is not None:
        args.camera = True  # Ensure camera flag is set if num_cameras is provided

    rclpy.init()
    
    try:
        if args.cvae:
            node = BCPolicyRunner_cvae(
                policy_path=args.policy,
                device=args.device,
                deterministic=args.deterministic,
                control_frequency=args.frequency,
                testing_mode=args.testing,
                context_length=args.context_length,
                camera_enabled=bool(args.camera),
                args_cli=args
            )
        else:
            node = BCPolicyRunner(
                policy_path=args.policy,
                device=args.device,
                deterministic=args.deterministic,
                control_frequency=args.frequency,
                testing_mode=args.testing,
                context_length=args.context_length,
                camera_enabled=bool(args.camera),
                args_cli=args
            )
        
        signal.signal(signal.SIGINT, lambda sig, frame: node.request_shutdown())
        rclpy.spin(node)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.try_shutdown()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()