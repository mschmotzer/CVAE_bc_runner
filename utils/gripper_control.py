import time
import numpy as np
import rclpy
from franka_msgs.action import Homing, Move, Grasp
from rclpy.action import ActionClient

class GripperControlMixin: 

    # Define gripper control methods
    def home_gripper(self):
        goal_msg = Homing.Goal()
        # Send goal async and forget (or handle future if needed)
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open' # Assume homing opens the gripper

    def open_gripper(self):
        """Open the gripper using the action client with safety checks"""
        with self.gripper_action_lock:
            current_time = time.time()
            
            # Safety checks
            if self.gripper_action_in_progress:
                self.get_logger().debug("Gripper action in progress, skipping open command")
                return
            
            if current_time - self.gripper_last_command_time < self.gripper_command_cooldown:
                self.get_logger().debug("Gripper cooldown active, skipping open command")
                return
                
            if self.gripper_goal_state == 'open':
                return  # Already open
                
            # Set state and send command
            self.gripper_action_in_progress = True
            self.gripper_last_command_time = current_time
        
        try:
            goal_msg = Move.Goal()
            goal_msg.width = self.gripper_max_width
            goal_msg.speed = self.gripper_speed
            
            # Send goal with result callback
            goal_future = self.move_client.send_goal_async(goal_msg)
            goal_future.add_done_callback(self._gripper_goal_callback)
            
            self.gripper_goal_state = 'open'
            self.get_logger().debug("Safe gripper OPEN command sent")
            
        except Exception as e:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False
            self.get_logger().error(f"Failed to send gripper open command: {e}")

    def close_gripper(self):
        """Close the gripper using the action client with safety checks"""
        with self.gripper_action_lock:
            current_time = time.time()
            
            # Safety checks
            if self.gripper_action_in_progress:
                self.get_logger().debug("Gripper action in progress, skipping close command")
                return
            
            if current_time - self.gripper_last_command_time < self.gripper_command_cooldown:
                self.get_logger().debug("Gripper cooldown active, skipping close command")
                return
                
            if self.gripper_goal_state == 'closed':
                return  # Already closed
                
            # Set state and send command
            self.gripper_action_in_progress = True
            self.gripper_last_command_time = current_time
        
        try:
            goal_msg = Grasp.Goal()
            goal_msg.width = 0.0
            goal_msg.speed = self.gripper_speed
            goal_msg.force = self.gripper_force
            goal_msg.epsilon.inner = self.gripper_epsilon_inner
            goal_msg.epsilon.outer = self.gripper_epsilon_outer

            # Send goal with result callback
            goal_future = self.grasp_client.send_goal_async(goal_msg)
            goal_future.add_done_callback(self._gripper_goal_callback)
            
            self.gripper_goal_state = 'closed'
            self.get_logger().debug("Safe gripper CLOSE command sent")
            
        except Exception as e:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False
            self.get_logger().error(f"Failed to send gripper close command: {e}")

    def _gripper_goal_callback(self, future):
        """Minimal callback to reset gripper action state"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                # Get result to reset state when complete
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self._gripper_result_callback)
            else:
                with self.gripper_action_lock:
                    self.gripper_action_in_progress = False
        except Exception:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False

    def _gripper_result_callback(self, future):
        """Reset gripper state when action completes"""
        with self.gripper_action_lock:
            self.gripper_action_in_progress = False

    def detect_gripper_state_change(self): 
        if np.abs(self.current_gripper_positions[0]) + np.abs(self.current_gripper_positions[1]) > self.release_threshold:
            return 'open'
        else:
            return 'closed'
    
    def toggle_gripper_manual(self):
        """Manually toggle gripper state between open and closed"""
        try:
            if self.gripper_goal_state == 'open' or self.gripper_goal_state == 'unknown':
                # Close the gripper
                self.close_gripper()
                print("Manual gripper command: CLOSING")
                self.update_status("Manual gripper: CLOSING", " - Press O again to open")
                
            elif self.gripper_goal_state == 'closed':
                # Open the gripper
                self.open_gripper()
                print("Manual gripper command: OPENING")
                self.update_status("Manual gripper: OPENING", " - Press O again to close")
                
        except Exception as e:
            self.get_logger().error(f"Error in manual gripper control: {e}")
            print(f"Manual gripper control failed: {e}")
    
    def handle_cube_attachment(self, gripper_state_change: str): 
        """
        Handle cube attachment and detachment based on gripper state and proximity
        - If gripper just closed, check proximity to determine which cube was grasped
        - If gripper just opened, detach the currently attached cube
        - Uses proximity threshold to determine if gripper is near a cube
        """
        
            # First Grasp "cube_2" (Red)
        grasp_width = np.abs(self.current_gripper_positions).sum()
        grasped = self.grasp_min < grasp_width and grasp_width < self.grasp_max 
        if grasped and self.counter == 0:
            self.ee_pos_cube_vicinity = np.array([self.current_eef_pose.pose.position.x,
                               self.current_eef_pose.pose.position.y,
                               self.current_eef_pose.pose.position.z])
        
        if self.grasp_sequence_count == 0: 
            # Check if gripper is near cube_2
            if grasped and self.counter == self.gripper_inrange:
                # Attach cube_2 and update count
                self.cube_attached = 'cube_2'
                self.grasp_sequence_count = 1
                print(f"🔴 CUBE_2 (Red) ATTACHED to gripper - Dynamic tracking enabled")
                self.get_logger().info("Cube_2 attached - enabling dynamic position tracking")
                self.cube_ee_offset = self.cube_positions['cube_2'] - self.ee_pos_cube_vicinity
                self.cube_ee_offset[1]=0.0  # Lock Y axis for cube_2
            elif grasped:
                self.counter += 1
            else:
                self.counter = 0
        # Second Grasp "cube_3" (Green)
        elif self.grasp_sequence_count == 1 and self.cube_attached is None:
            # Check if gripper is near cube_3
            if grasped and self.counter == self.gripper_inrange:
                self.cube_attached = 'cube_3'
                self.grasp_sequence_count = 2
                print(f"🟢 CUBE_3 (Green) ATTACHED to gripper - Dynamic tracking enabled")
                self.get_logger().info("Cube_3 attached - enabling dynamic position tracking")
                self.cube_ee_offset = self.cube_positions['cube_3'] - self.ee_pos_cube_vicinity
                self.cube_ee_offset[1]=0.0  # Lock X axis for cube_3

            elif grasped:
                self.counter += 1
            else:
                self.counter = 0

    def is_gripper_near_cube(self, cube_name: str) -> bool: 
        """ Helper function: Check if gripper is close enough to cube to grasp it"""
        # Get current ee position
        ee_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])

        # Get cube position -> self.cube_positions is a dict with cube names as keys and positions as values
        cube_pos = self.cube_positions[cube_name]

        # Calculate distance 
        distance = np.linalg.norm(ee_pos - cube_pos)

        # Return True if within proximity threshold, False otherwise
        return distance < self.proximity_threshold


    # Helper to wait for action servers
    def wait_for_action_server(self, client, name):
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
            self.get_logger().info(f'{name} action server not available, waiting again...')
        if rclpy.ok():
            self.get_logger().info(f'{name} action server found.')
        else:
             self.get_logger().error(f'ROS shutdown while waiting for {name} server.')
             raise SystemExit('ROS shutdown')
    

    def initialize_gripper_clients(self):
        """Initialize all gripper action clients"""
        self.homing_client = ActionClient(self, Homing, '/fr3_gripper/homing', callback_group=self.callback_group)
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move', callback_group=self.callback_group)
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp', callback_group=self.callback_group)
        
        # Wait for all servers
        self.wait_for_action_server(self.homing_client, 'Homing')
        self.wait_for_action_server(self.move_client, 'Move')
        self.wait_for_action_server(self.grasp_client, 'Grasp')
        
        # Perform initial homing
        self.home_gripper()