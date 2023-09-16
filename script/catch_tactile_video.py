#!/usr/bin/env python3
import time
from copy import deepcopy

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from tactile_adaptive_grasp.image_flow import Flow


class CatchVideo(Node):
    def __init__(self):
        super().__init__('gym_pipe')
        self.declare_parameter('control_frequency', 5.0)
        self.declare_parameter("base_link", "link_0")
        self.declare_parameter("end_effector_link", "link_ee")
        self.declare_parameter("end_effector_link_expect", "link_ee_expect")

        self.cv_bridge = CvBridge()
        self.left_subscription_rgb = self.create_subscription(Image, '/gsmini_left/rgb', self.left_rgb_cb_, 5)
        self.right_subscription_rgb = self.create_subscription(Image, '/gsmini_right/rgb', self.right_rgb_cb_, 5)
        self.left_subscription_depth = self.create_subscription(Image, '/gsmini_left/depth', self.left_depth_cb_, 5)
        self.right_subscription_depth = self.create_subscription(Image, '/gsmini_right/depth', self.right_depth_cb_, 5)
        self.left_opt_flow = Flow(col=320, row=240)
        self.right_opt_flow = Flow(col=320, row=240)
        self.left_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.right_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.left_depth = np.zeros((240, 320), dtype=np.uint8)
        self.right_depth = np.zeros((240, 320), dtype=np.uint8)

        output_file = '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5.0
        frame_size = (320, 240)
        self.left_rgb_video_writer = cv2.VideoWriter('left_rgb'+output_file, fourcc, fps, frame_size)
        self.right_rgb_video_writer = cv2.VideoWriter('right_rgb' + output_file, fourcc, fps, frame_size)
        self.left_depth_video_writer = cv2.VideoWriter('left_dep'+output_file, fourcc, fps, frame_size,isColor=False)
        self.right_depth_video_writer = cv2.VideoWriter('right_dep' + output_file, fourcc, fps, frame_size,isColor=False)

    def left_rgb_cb_(self, msg: Image):
        self.left_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        flow=self.left_opt_flow.get_flow(self.left_rgb)
        self.left_opt_flow.draw(self.left_rgb,flow)
        self.left_rgb_video_writer.write(self.left_rgb)

    def right_rgb_cb_(self, msg: Image):
        self.right_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        flow = self.right_opt_flow.get_flow(self.right_rgb)
        self.right_opt_flow.draw(self.right_rgb,flow)
        self.right_rgb_video_writer.write(self.right_rgb)

    def left_depth_cb_(self, msg: Image):
        self.left_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # depth_c3=cv2.cvtColor(self.left_depth,cv2.COLOR_GRAY2BGR)
        self.left_depth_video_writer.write(self.left_depth)

    def right_depth_cb_(self, msg: Image):
        self.right_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # depth_c3=cv2.cvtColor(self.right_depth,cv2.COLOR_GRAY2BGR)
        self.right_depth_video_writer.write(self.right_depth)

    #
    # def get_total_and_center(depth_image: np.ndarray):
    #     M = cv2.moments(depth_image)
    #     if M["m00"] <= 1e-6:
    #         return [0, 120, 160]
    #     else:
    #         cx = M["m10"] / M["m00"]
    #         cy = M["m01"] / M["m00"]
    #         return M['m00'] / 255 / 240 / 320, cx, cy


def main(args=None):
    rclpy.init(args=args)
    node = CatchVideo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
