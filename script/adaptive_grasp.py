#!/usr/bin/env python3
import re
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import cv2
from rclpy.node import Node
from bye140_msg.srv import CalibrateGripper, MoveTo, GetCalibrated
from bye140_msg.msg import ByStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AdaptiveGrasp(Node):
    def __init__(self):
        super().__init__('bye140_node')
        self.declare_parameter('control_frequency', 5.0)
        self.left_subscription = self.create_subscription(Image, '/gsmini_left/depth', self.left_depth_cb_, 5)
        self.right_subscription = self.create_subscription(Image, '/gsmini_right/depth', self.left_depth_cb_, 5)
        self.gripper_status = self.create_subscription(ByStatus, '/by_status', self.gripper_status_cb, 10)
        self.timer_callback_group=MutuallyExclusiveCallbackGroup()
        self.gripper_move = self.create_client(MoveTo, '/moveto')
        self.control_timer = self.create_timer(1 / self.get_parameter('control_frequency').value,
                                               self.gripper_move_command,callback_group=self.timer_callback_group)

        self.cv_bridge = CvBridge()
        self.left_depth = np.zeros((240, 320), dtype=np.uint8)
        self.right_depth = np.zeros((240, 320), dtype=np.uint8)
        self.left_feature = np.zeros(3, dtype=np.float32)
        self.right_feature = np.zeros(3, dtype=np.float32)

        self.gripper_pos = 0.0
        self.command_pos = 30.0

        self.tactile_calibrate_flag=0

    def left_depth_cb_(self, msg: Image):
        self.get_logger().info(f"into depth cb")
        self.left_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        m0, cx, cy = get_total_and_center(self.left_depth)
        self.left_feature[0],self.left_feature[1],self.left_feature[2]=m0,cx,cy
        self.get_logger().info(f"depth_total:{m0}")
        left_depth_c3 = cv2.cvtColor(self.left_depth, cv2.COLOR_GRAY2BGR)
        left_depth_c3 = cv2.circle(left_depth_c3, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        # cv2.imshow("center", left_depth_c3)
        # cv2.waitKey(1)
        self.tactile_calibrate_flag+=1
        self.get_logger().info("got here")

    def right_depth_cb(self, msg: Image):
        self.right_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        m0, cx, cy = get_total_and_center(self.right_depth)
        self.right_feature[0], self.right_feature[1], self.right_feature[2] = m0, cx, cy

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
        if self.left_feature[0]<0:
            self.command_pos = self.command_pos - 1
        else:
            self.command_pos = self.command_pos - (0.02-self.left_feature[0])*10

        if self.tactile_calibrate_flag<50:
            self.command_pos=30.0
        request = MoveTo.Request(position=self.command_pos, speed=150.0, acceleration=500.0,
                                 torque=0.2, tolerance=20.0, waitflag=False)
        self.get_logger().info("start service call")
        future = self.gripper_move.call(request)

        # if future.result() is True:
        #     self.get_logger().info('yes!')
        # else:
        #     self.get_logger().info('Service call failed.')


def get_total_and_center(depth_image: np.ndarray):
    M = cv2.moments(depth_image)
    if M["m00"] <= 1e-6:
        return [0, 120, 160]
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return M['m00']/255/240/320, cx, cy


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
