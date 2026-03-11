#!/usr/bin/env python3
"""
Policy Control Mixin for BC Policy Runner
Handles policy lifecycle management and robot state control.
"""
import numpy as np
import torch
from std_msgs.msg import Float64MultiArray
import sys
import sys
sys.path.append('/home/pdz/MasterThesis_MSC/Isaaclab_5/IsaacLab/')
from act_copy.policy_runner import ACTPolicy

class PolicyControlMixin_cvae:
    def cvae_policy_loader( policy_path: str, device: str = "cpu", verbose: bool = False, args_cli=None):
        """
        Load the trained BC policy using robomimic's policy_from_checkpoint.

        Args:
            policy_path (str): Path to the policy checkpoint (.pth)
            device (str): Device to load the policy on ("cpu" or "cuda")
            verbose (bool): Whether to print detailed loading info

        Returns:
            policy: The loaded policy object
            ckpt_dict: The checkpoint dictionary with metadata
    """

        lr_backbone = 1e-5
        backbone = 'resnet18'

        enc_layers = 6
        enc_dec_layers = 6
        dec_layers = 9
        nheads = 8
        policy_config = {   'ckpt_dir': policy_path,
                            'num_queries': 64,
                            'lr_backbone': lr_backbone,
                            'backbone': backbone,
                            'enc_layers': enc_layers,
                            'enc_decoder_layers': enc_dec_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': ["image", "image2"],
                            'velocity_control': args_cli.velocity_control,
                            'context_length': args_cli.context_length
                            }
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(args_cli.policy))

        return policy

        
    def start_policy(self):
        """Start the policy with proper LSTM reset"""
        if not self.is_running:
            self.is_running = True
            self.episode_active = True
            self.step_count = 0
 
            # Enable CSV recording
            self.policy_running = True
            
            self.update_status("Policy started", f"Control freq: {self.control_frequency}Hz")

    def stop_policy(self):
        """Stop the policy and reset for next episode"""
        if self.is_running:
            self.is_running = False
            self.episode_active = False
            
            # Disable CSV recording
            self.policy_running = False
            
             
            self.update_status("Policy stopped")

    def reset_to_home(self):
        """Reset robot to safe home position"""
        # Stop policy execution first
        was_running = self.is_running
        if self.is_running:
            self.stop_policy()
        
        print("Resetting robot to home position...")
        self.update_status("Status: HOMING - Moving to safe position...")
        
        try:
            # Define safe home position (adjust these values based on your robot setup)
            home_position = np.array([ 0.4728, -0.0671,  0.2552])  # Safe position above workspace
            home_quaternion_ros = np.array([1.0, 0.0, 0.0, 0.0])  # Pointing down [qx, qy, qz, qw]
            
            # Create cartesian pose command: [x, y, z, qx, qy, qz, qw]
            home_pose = np.concatenate([
                home_position,        # [x, y, z]
                home_quaternion_ros   # [qx, qy, qz, qw]
            ])
            
            # Publish home pose command
            pose_msg = Float64MultiArray()
            pose_msg.data = home_pose.tolist()
            self.pose_command_pub.publish(pose_msg)
            
            # Open gripper to safe state
            self.open_gripper()
            
            # Reset episode state
            self.object_grasped = False
            self.gripper_goal_state = 'open'
            
            # RESET CUBE ATTACHMENT STATE
            self.grasp_sequence_count = 0
            self.cube_attached = None
            self.last_gripper_state = 'open'
            self.obs_dict = None
            print("🔄 Cube attachment state reset")
            
            print("Robot moved to home position and episode state reset")

            # Print the instructions after each reset based on mode:
            if self.testing_mode:
                self.print_instructions_testing()
            else:
                self.print_instructions()
            
            # Update status based on previous running state
            if was_running:
                self.update_status(f"Status: HOMED - Trial {self.trial_id} ready - Press SPACE to resume, S to stop, Q to quit")
            else:
                self.update_status("Status: HOMED - Press SPACE to start, Q to quit")
                
            # Reset camera detection state
            if hasattr(self, 'reset_camera_detection'):
                self.reset_camera_detection()
                
        except Exception as e:
            self.get_logger().error(f"Error during home reset: {e}")
            self.update_status("Status: HOME FAILED - Check robot state")
    def execute_action(self, action_np: np.ndarray):
        """Unified action execution for both normal and replay modes"""
        try:
            # Ensure it's a 1D array
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            # Interpret action - 7D end-effector pose + 1D gripper
            eef_pose = action_np[:7]  # [x, y, z, qw, qx, qy, qz] - IsaacLab format
            gripper_command = action_np[7]  # Gripper command
                
            # Extract position and quaternion from pose
            position = eef_pose[:3]  # [x, y, z]
            quaternion_sim = eef_pose[3:]  # [qw, qx, qy, qz] - IsaacLab format
            
            # Convert from IsaacLab [qw, qx, qy, qz] to ROS [qx, qy, qz, qw]
            quaternion_ros = np.array([
                quaternion_sim[1],  # qx
                quaternion_sim[2],  # qy
                quaternion_sim[3],  # qz
                quaternion_sim[0]   # qw
            ])
            
            # Normalize quaternion to ensure it's valid
            quat_norm = np.linalg.norm(quaternion_ros)
            if quat_norm > 0:
                quaternion_ros = quaternion_ros / quat_norm
            else:
                self.get_logger().warn("Invalid quaternion, skipping action execution")
                return
            
            # --- Gripper Control Logic ---
            # Clamp gripper command to expected range [-1, 1]
            if gripper_command < 0.25:
                #self.close_gripper()
                gripper_command = -1.0
            else:
                #self.open_gripper()
                gripper_command = 1.0
            """gripper_command = np.clip(gripper_command, -0.25, 0.25)
            gripper_command = np.clip(gripper_command, -0.5, 0.5)
            """
            
            # Determine desired gripper state based on command
            desired_gripper_state = 'closed' if gripper_command < 0 else 'open'
            if desired_gripper_state != self.gripper_goal_state:
                if desired_gripper_state == 'open':
                    self.open_gripper()
                elif desired_gripper_state == 'closed':
                    self.close_gripper()
            
            # Create cartesian pose command: [x, y, z, qx, qy, qz, qw]
            cartesian_pose = np.concatenate([
                position,         # [x, y, z]
                quaternion_ros    # [qx, qy, qz, qw]
            ])
            
            # Publish cartesian pose commands to the controller
            pose_msg = Float64MultiArray()
            pose_msg.data = cartesian_pose.tolist()
            self.pose_command_pub.publish(pose_msg)

            self.get_logger().debug(f"Action executed: pos={position}, quat={quaternion_ros}, gripper={gripper_command}")
            
        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")