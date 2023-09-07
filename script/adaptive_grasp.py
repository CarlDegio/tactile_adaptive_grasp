#!/usr/bin/env python3
import re
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import cv2
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from bye140_msg.srv import MoveTo
from bye140_msg.msg import ByStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tactile_adaptive_grasp.image_flow import Flow


class AdaptiveGrasp(Node):
    def __init__(self):
        super().__init__('adaptive_grasp_node')
        self.declare_parameter('control_frequency', 5.0)
        self.left_subscription_depth = self.create_subscription(Image, '/gsmini_left/depth', self.left_depth_cb_, 5)
        self.right_subscription_depth = self.create_subscription(Image, '/gsmini_right/depth', self.left_depth_cb_, 5)
        self.left_subscription_rgb = self.create_subscription(Image, '/gsmini_left/rgb', self.left_rgb_cb_, 5)
        self.right_subscription_rgb = self.create_subscription(Image, '/gsmini_right/rgb', self.right_rgb_cb_, 5)
        self.left_feature_pub = self.create_publisher(Float32MultiArray, '/left_feature', 10)
        self.right_feature_pub = self.create_publisher(Float32MultiArray, '/right_feature', 10)

        self.gripper_status = self.create_subscription(ByStatus, '/by_status', self.gripper_status_cb, 10)
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.gripper_move = self.create_client(MoveTo, '/moveto')
        # self.control_timer = self.create_timer(1 / self.get_parameter('control_frequency').value,
        #                                        self.gripper_move_command, callback_group=self.timer_callback_group)
        self.reset_gripper = self.create_service(Trigger, '/slack_gripper', self.slack_gripper)
        self.start_gripper = self.create_service(Trigger, '/start_gripper', self.start_gripper)

        self.cv_bridge = CvBridge()
        self.left_opt_flow = Flow(col=320, row=240)
        self.right_opt_flow = Flow(col=320, row=240)
        self.left_depth = np.zeros((240, 320), dtype=np.uint8)
        self.left_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.right_depth = np.zeros((240, 320), dtype=np.uint8)
        self.right_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.left_feature = np.zeros(5, dtype=np.float32)
        self.right_feature = np.zeros(5, dtype=np.float32)
        self.dimension = MultiArrayDimension()
        self.dimension.label = 'tactile_feature'
        self.dimension.size = 5
        self.dimension.stride = 4

        self.gripper_pos = 0.0
        self.command_pos = 30.0

        self.tactile_calibrate_flag = 0

    def left_depth_cb_(self, msg: Image):
        self.left_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        m0, cx, cy = get_total_and_center(self.left_depth)
        self.left_feature[0], self.left_feature[1], self.left_feature[2] = m0, cx, cy

        # 为了数据同步
        self.left_opt_flow.get_flow(self.left_rgb)
        self.left_feature[3], self.left_feature[4] = self.left_opt_flow.get_flow_entropy()

        self.get_logger().info("left_feature" + str(self.left_feature))
        # left_depth_c3 = cv2.cvtColor(self.left_depth, cv2.COLOR_GRAY2BGR)
        # left_depth_c3 = cv2.circle(left_depth_c3, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        # cv2.imshow("center", left_depth_c3)
        # cv2.waitKey(1)
        feature_msg = Float32MultiArray()
        feature_msg.layout.dim.append(self.dimension)
        feature_msg.data = self.right_feature.tolist()
        self.left_feature_pub.publish(feature_msg)

    def left_rgb_cb_(self, msg: Image):
        self.left_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def right_depth_cb(self, msg: Image):
        self.right_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        m0, cx, cy = get_total_and_center(self.right_depth)
        self.right_feature[0], self.right_feature[1], self.right_feature[2] = m0, cx, cy

        self.right_opt_flow.get_flow(self.right_rgb)
        self.right_feature[3], self.right_feature[4] = self.right_opt_flow.get_flow_entropy()

        feature_msg = Float32MultiArray()
        feature_msg.layout.dim.append(self.dimension)
        feature_msg.data = self.right_feature.tolist()
        self.right_feature_pub.publish(feature_msg)

    def right_rgb_cb_(self, msg: Image):
        self.right_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def gripper_status_cb(self, msg: ByStatus):
        self.gripper_pos = msg.position

    def gripper_move_command(self):
        """
        float64 position
        float64 speed
        float64 acceleration
        float64 torque
        float64 tolerance
        bool waitflag
        :return:
        """
        if self.left_feature[0] < 0:
            self.command_pos = self.command_pos - 1
        else:
            self.command_pos = self.command_pos - (0.02 - self.left_feature[0]) * 10

        self.tactile_calibrate_flag += 1
        if self.tactile_calibrate_flag < 50:
            self.command_pos = 30.0
        request = MoveTo.Request(position=self.command_pos, speed=150.0, acceleration=500.0,
                                 torque=0.2, tolerance=20.0, waitflag=False)
        future = self.gripper_move.call(request)

        # if future.result() is True:
        #     self.get_logger().info('yes!')
        # else:
        #     self.get_logger().info('Service call failed.')

    def slack_gripper(self):
        self.tactile_calibrate_flag = 0
        self.control_timer.cancel()

    def start_gripper(self):
        self.control_timer.reset()


def get_total_and_center(depth_image: np.ndarray):
    M = cv2.moments(depth_image)
    if M["m00"] <= 1e-6:
        return [0, 120, 160]
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return M['m00'] / 255 / 240 / 320, cx, cy


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveGrasp()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
