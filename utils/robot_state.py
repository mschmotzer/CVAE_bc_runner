#!/usr/bin/env python3
"""
Robot State Mixin for BC Policy Runner
Handles all robot sensor callbacks and state management.
"""
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import torch
from sensor_msgs.msg import Image

class RobotStateMixin:
    """Mixin class for handling robot sensor callbacks and state management"""
    
    def eef_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_eef_pose = msg  # Quaternion is in x, y, z, w format

    def jacobian_callback(self, msg: Float64MultiArray):
        """Callback for jacobian updates with debugging"""
        if len(msg.data) != 42:  # 6x7 = 42 elements
            return
        
        # Reshape Jacobian (column-major order as published by the controller)
        self.current_jacobian = np.array(msg.data).reshape((6, 7), order='F')


    def image1_callback(self, msg: Image):
        """Callback for image1 updates with visualization"""

        # Convert ROS Image -> numpy
        img1_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # Convert to torch tensor
        self.img1 = torch.from_numpy(img1_np).float()

        # Convert RGB -> BGR for OpenCV display
        img1_vis = cv2.cvtColor(img1_np, cv2.COLOR_RGB2RGBA)

        # Show image
        cv2.imshow("Camera 1", img1_vis)
        cv2.waitKey(1)


    def image2_callback(self, msg: Image):
        """Callback for image2 updates with visualization"""

        # Convert ROS Image -> numpy
        img2_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # Convert to torch tensor
        self.img2 = torch.from_numpy(img2_np).float()

        # Convert RGB -> BGR for OpenCV display
        img2_vis = cv2.cvtColor(img2_np, cv2.COLOR_RGB2RGBA)

        # Show image
        cv2.imshow("Camera 2", img2_vis)
        cv2.waitKey(1)



    def joint_state_callback(self, msg: JointState):
        msg_positions = np.array(msg.position)
        msg_velocities = np.array(msg.velocity)
        self.joint_pos = np.zeros(len(msg_positions))
        self.joint_vel = np.zeros(len(msg_velocities))
        self.joint_pos[0] = msg_positions[0]
        self.joint_pos[1] = msg_positions[4]
        self.joint_pos[2] = msg_positions[1]
        self.joint_pos[3] = msg_positions[5]
        self.joint_pos[4] = msg_positions[6]
        self.joint_pos[5] = msg_positions[2]
        self.joint_pos[6] = msg_positions[3]
        #self.joint_pos -= [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952]
        self.joint_vel[0] = msg_velocities[0]
        self.joint_vel[1] = msg_velocities[4]
        self.joint_vel[2] = msg_velocities[1]
        self.joint_vel[3] = msg_velocities[5]
        self.joint_vel[4] = msg_velocities[6]
        self.joint_vel[5] = msg_velocities[2]
        self.joint_vel[6] = msg_velocities[3]
    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        # Gripper should have symmetric but opposite values: [+value, -value] "IsaacLab Convention"
        

        finger_1_pos = msg.position[0]  # First finger (positive)
        finger_2_pos = -msg.position[1] # Second finger (negative)
        finger_vel_1 = msg.velocity[0]
        finger_vel_2 = -msg.velocity[1]
        self.prev_gripper = np.array([finger_1_pos, finger_2_pos])
        self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])
        self.current_gripper_velocities = np.array([finger_vel_1, finger_vel_2])

    def calculate_manipulability_index(self):
        """Calculate Yoshikawa's manipulability measure: sqrt(det(J * J^T))"""
        if self.current_jacobian is None:
            return 0.0
        
        try:
            # Verify Jacobian data
            if not np.isfinite(self.current_jacobian).all():
                self.get_logger().warn("Jacobian contains NaN or infinite values")
                return 0.0
            
            # Yoshikawa's manipulability: sqrt(det(J * J^T))
            # This gives the volume of the manipulability ellipsoid
            JJT = self.current_jacobian @ self.current_jacobian.T  # 6x6 matrix
            det_JJT = np.linalg.det(JJT)
            
            # Take square root and ensure non-negative
            manipulability_index = np.sqrt(max(0.0, det_JJT))
            
            return float(manipulability_index)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating manipulability index: {e}")
            return 0.0
        
        
    def cube_poses_callback(self, msg):
        """Callback for camera-detected cube poses"""
        try:
            # Expect 9 elements: [cube_1_x, cube_1_y, cube_1_z, cube_2_x, cube_2_y, cube_2_z, cube_3_x, cube_3_y, cube_3_z]
            if len(msg.data) != 9:
                self.get_logger().warn(f"Expected 9 elements for 3 cube positions, got {len(msg.data)}")
                return
            
            # Extract positions for each cube
            cube_1_position = np.array([msg.data[0], msg.data[1], msg.data[2]])
            cube_2_position = np.array([msg.data[3], msg.data[4], msg.data[5]])
            cube_3_position = np.array([msg.data[6], msg.data[7], msg.data[8]])
            
            # Identity quaternion for all cubes [w, x, y, z] = [1, 0, 0, 0] in IsaacLab format
            identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Store camera-detected poses for all cubes
            self.camera_cube_poses = {
                'cube_1': {
                    'position': cube_1_position,
                    'quaternion': identity_quaternion
                },
                'cube_2': {
                    'position': cube_2_position, 
                    'quaternion': identity_quaternion
                },
                'cube_3': {
                    'position': cube_3_position,
                    'quaternion': identity_quaternion
                }
            }
            
            self.camera_poses_received = True
            
            self.get_logger().debug(f"Received cube poses from camera:")
            self.get_logger().debug(f"  Cube 1: {cube_1_position}")
            self.get_logger().debug(f"  Cube 2: {cube_2_position}")
            self.get_logger().debug(f"  Cube 3: {cube_3_position}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing cube poses from camera: {e}")
    
    def setup_camera_subscribers(self, qos_profile, callback_group):
        """Setup subscribers for camera-detected cube poses"""
        # Initialize camera data storage
        self.camera_cube_poses = {}
        self.camera_poses_received = False
        self.expected_cubes = ['cube_1', 'cube_2', 'cube_3']
        self.received_cubes = set()
        
        # Create a single subscriber to the unified topic
        self.cube_subscriber = self.create_subscription(
            PoseStamped,
            '/perception/object_pose',
            self.cube_pose_callback,
            qos_profile,
            callback_group=callback_group
        )
        
        self.get_logger().info("Camera cube pose subscriber initialized for /perception/object_pose")

    def cube_pose_callback(self, msg):
        """Callback for unified cube pose messages from /perception/object_pose"""
        try:
            # Extract cube identity from frame_id
            # Expected format: "panda_link0_cube_X_idY" where X is cube number (1,2,3)
            frame_id = msg.header.frame_id
            
            # Parse frame_id to extract cube number
            cube_name = None
            if "cube_1" in frame_id:
                cube_name = "cube_1"
            elif "cube_2" in frame_id:
                cube_name = "cube_2"
            elif "cube_3" in frame_id:
                cube_name = "cube_3"
            else:
                self.get_logger().warn(f"Unknown frame_id format: {frame_id}")
                return
            
            # Extract position from PoseStamped message
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            
            # Use identity quaternion [w, x, y, z] = [1, 0, 0, 0] in IsaacLab format
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Store camera-detected pose for this cube
            self.camera_cube_poses[cube_name] = {
                'position': position,
                'quaternion': quaternion
            }
            
            # Track which cubes we've received
            self.received_cubes.add(cube_name)
            
            # Mark as received if we have all cubes
            if len(self.received_cubes) >= len(self.expected_cubes):
                self.camera_poses_received = True
            
            self.get_logger().debug(f"Received pose for {cube_name} from frame_id {frame_id}: pos={position}, quat={quaternion}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing unified cube pose: {e}")

    # -------------------------------Cube Pose Callbacks-------------------------------------------------
    def redcube_callback(self, msg: PoseStamped):
        """Callback for red cube pose updates"""
        try:
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z+0.012
            ])
            orientation = np.array([
                
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            self.cube_positions['cube_2'] = position
            self.cube_quaternions['cube_2'] = orientation
        except Exception as e:
            self.get_logger().error(f"Red cube callback error: {e}")

    def bluecube_callback(self, msg: PoseStamped):
        """Callback for blue cube pose updates"""
        try:
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z+0.012
            ])
            orientation = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            self.cube_positions['cube_1'] = position
            self.cube_quaternions['cube_1'] = orientation
        except Exception as e:
            self.get_logger().error(f"Blue cube callback error: {e}")

    def greencube_callback(self, msg: PoseStamped):
        """Callback for green cube pose updates"""
        try:
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z+0.012
            ])
            orientation = np.array([
                
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])
            self.cube_positions['cube_3'] = position
            self.cube_quaternions['cube_3'] = orientation
        except Exception as e:
            self.get_logger().error(f"Green cube callback error: {e}")