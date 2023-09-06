#!/usr/local/bin/python
# encoding: utf-8
# File Name: optical_flow.py
# Author: Shaoxiong (Shawn) Wang
# Create Time: 2017/11/15 16:24
# TODO:
# Add annotation
import cv2
import numpy as np

from scipy.stats import entropy

"""
光流法显示标记点的运动趋势
"""


class Flow:
    def __init__(self, col, row):
        self.x = np.zeros((row, col)).astype(int)
        self.y = np.zeros((row, col)).astype(int)
        self.col = col
        self.row = row
        self.prvs_gray = np.zeros((row, col)).astype(np.uint8)
        self.opt_flow = np.zeros((row, col, 2)).astype(np.float32)
        self.initial = False

    def get_flow(self, img):
        if not self.initial:
            self.initial = True
            self.prvs_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.opt_flow = cv2.calcOpticalFlowFarneback(self.prvs_gray, next_gray, None, 0.5, 3, int(180 * self.col / 960),
                                                     5, 5, 1.2, 0)
        return self.opt_flow

    def get_flow_entropy(self):
        flow = self.opt_flow.reshape(-1, 2)
        histx, _ = np.histogram(flow[:, 0], bins=20, range=(-7, 7))
        histy, _ = np.histogram(flow[:, 1], bins=20, range=(-7, 7))
        probabilities_x = histx / len(flow)
        entropy_value_x = entropy(probabilities_x)
        probabilities_y = histy / len(flow)
        entropy_value_y = entropy(probabilities_y)
        return entropy_value_x, entropy_value_y

    def draw(self, img, flow, scale=5.0):  # draw the arrowedline in the image
        start_line = 40
        start_vertical = 40
        step = 20
        d_all = np.zeros((round((self.row - start_vertical) / step), round((self.col - start_line) / step), 2))
        m, n = 0, 0
        for i in range(start_vertical, self.row, step):
            for j in range(start_line, self.col, step):
                d = (flow[i, j] * scale).astype(int)
                cv2.arrowedLine(img, (j, i), (j + d[0], i + d[1]), (0, 255, 255),
                                1)  # cv2.arrowedLine(img, startpoint(x,y), endpoint(x,y), color, linedwidth)
                d_all[m, n] = d / scale
                n += 1
            m += 1
            n = 0

        return d_all

    def draw_sumline(self, img, center, sum_x, sum_y, scale=5.0):
        height = img.shape[0]
        width = img.shape[1]
        cv2.arrowedLine(img, (int(center[0]), int(center[1])),
                        (int(center[0] + sum_x * scale), int(center[1] + sum_y * scale)),
                        (0, 0, 255), 2)