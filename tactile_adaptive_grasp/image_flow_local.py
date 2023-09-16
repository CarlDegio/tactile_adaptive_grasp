#!/usr/local/bin/python
# encoding: utf-8
# File Name: optical_flow.py
# Author: Shaoxiong (Shawn) Wang
# Create Time: 2017/11/15 16:24
# TODO:
# Add annotation
import cv2
import numpy as np
import numpy.matlib
import time

from scipy.stats import entropy

"""
光流法显示标记点的运动趋势
"""

class Flow:
    def __init__(self, col, row):
        x0 = np.matlib.repmat(np.arange(row), col, 1).T
        y0 = np.matlib.repmat(np.arange(col), row, 1)

        self.x = np.zeros_like(x0).astype(int)
        self.y = np.zeros_like(y0).astype(int)
        self.x0 = x0
        self.y0 = y0
        self.col = col
        self.row = row

    def get_raw_img(self, frame):
        img = cv2.resize(frame, (895, 672))  # size suggested by janos to maintain aspect ratio
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
            np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (self.col, self.row))  # final resize for 3d

        return img

    def add_flow(self, flow):
        dx = np.round(self.x + self.x0).astype(int)
        dy = np.round(self.y + self.y0).astype(int)
        dx[dx >= self.row] = self.row - 1
        dx[dx < 0] = 0
        dy[dy >= self.col] = self.col - 1
        dy[dy < 0] = 0
        ds = np.reshape(flow[np.reshape(dx, -1), np.reshape(dy, -1)], (self.row, self.col, -1))
        nx = self.x + ds[:, :, 0]
        ny = self.y + ds[:, :, 1]
        return nx, ny

    def flow2color(self, flow, hsv, K=15):  # make the flow vector into the rgb color of each pixel
        mag, ang = cv2.cartToPolar(-flow[..., 1], flow[..., 0])
        hsv[..., 0] = ang * 180 / np.pi / 2
        mag = mag.astype(float) * K * 960 / self.col

        mag[mag > 255] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = mag
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

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

    def cal_force(self, img, flow):  # claculate the trendlines' vector sum to represent the force in the paralleral and the torque of the z axis.
        force_x = 0
        force_y = 0
        torque = 0  # the center when calculating the torque should be reset at the center of the object.
        height = img.shape[0]
        width = img.shape[1]
        for i in range(len(flow)):
            for j in range(len(flow[0])):
                force_x += flow[i, j][0]
                force_y += flow[i, j][1]
                torque += flow[i, j][0] * (height / 2 - i) + flow[i, j][1] * (j - width / 2)

        return force_x, force_y, torque

    def draw_sumLine(self, img, center, sum_x, sum_y, scale=5.0):
        height = img.shape[0]
        width = img.shape[1]
        cv2.arrowedLine(img, (int(center[0]), int(center[1])), (int(center[0] + sum_x * scale), int(center[1] + sum_y * scale)),
                        (0, 0, 255), 2)


if __name__ == "__main__":
    # cap = cv2.VideoCapture("first time/during.avi") # choose to read from video or camera
    # finger = gsdevice.Finger.MINI
    # cam_id = 0
    # dev = gsdevice.Camera(finger, cam_id)
    # cap = dev.connect()
    # 第一个传感器
    cap = cv2.VideoCapture(2)

    # 第二个传感器

    Flow = Flow(col=320, row=240)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    col = 320
    row = 240
    out = cv2.VideoWriter('flow.avi', fourcc, 3.5, (col * 1, row * 1))  # The fps depends on CPU
    out_2 = cv2.VideoWriter('flow_init.avi', fourcc, 3.5, (col * 1, row * 1))

    time.sleep(1)
    # 第一个传感器
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (col, row))
    # frame1 = dev.get_raw_image()
    frame1 = Flow.get_raw_img(frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # 第二个传感器

    flow_sum = np.zeros((Flow.row, Flow.col, 2))
    flow_trend = np.zeros(0)
    count = 0
    d = {}

    reset_threshold_error = 0.3  # ?
    reset_threshold_mean = 2  # ?

    while (1):
        count += 1
        # 第一个传感器
        try:
            ret, frame2 = cap.read()
            # frame2 = dev.get_raw_image()
            frame2 = Flow.get_raw_img(frame2)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
        except:
            break

        flow = cv2.calcOpticalFlowFarneback(f0, next, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
        flow_flatten = flow.reshape(-1, 2)
        histx, _ = np.histogram(flow_flatten[:, 0], bins=20, range=(-7, 7))
        histy, _ = np.histogram(flow_flatten[:, 1], bins=20, range=(-7, 7))
        probabilities_x = histx / len(flow_flatten)
        entropy_value_x = entropy(probabilities_x)
        probabilities_y = histy / len(flow_flatten)
        entropy_value_y = entropy(probabilities_y)
        print("entropy:",entropy_value_x,entropy_value_y)
        # bgr0 = Flow.flow2color(flow,hsv, K=100)
        # flow_2 = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, int(180 * Flow.col / 960), 5, 5, 1.2, 0)
        # bgr2 = Flow.flow2color(flow_2,hsv, K=100)

        # nx, ny = Flow.add_flow(flow_2)
        # nx = nx
        # ny = ny
        # error = (np.mean(np.abs(nx - flow[:, :, 0])) + np.mean(np.abs(ny - flow[:, :, 1]))) / 2.0
        # mean = (np.mean(np.abs(flow[:, :, 0])) + np.mean(np.abs(flow[:, :, 1]))) / 2.0

        # if error < reset_threshold_error or mean < reset_threshold_mean:
        #     Flow.x = flow[:, :, 0]
        #     Flow.y = flow[:, :, 1]
        # else:
        #     Flow.x, Flow.y = nx, ny

        # x, y = add_flow(x, y, flow_2)
        # flow_sum[:, :, 0] = Flow.x
        # flow_sum[:, :, 1] = Flow.y
        flow_sum[:, :, 0] = flow[:, :, 0]
        flow_sum[:, :, 1] = flow[:, :, 1]
        # bgr = Flow.flow2color(flow_sum, hsv)

        frame3 = np.copy(frame2)
        # frame4 = np.copy(frame2)
        # draw(frame2, flow_2) #frame2是此刻图像与上一时刻图像的光流图
        d_flow = Flow.draw(frame3, flow_sum)  # frame3是此刻与初始时刻的光流图（校正后）
        # flow_x, flow_y, flow_t = Flow.cal_force(frame3, d_flow)
        # Flow.draw_sumLine(frame3, flow_x, flow_y)
        # print("flow_x= ", flow_x)
        # print("flow_y= ", flow_y)
        # flow_trend = np.append(flow_trend, flow_y)
        # draw(frame4, flow) #frame4是此刻与初始时刻的光流图（未校正）

        # bgr = np.vstack([np.hstack([frame2, frame4, frame3]), np.hstack([bgr2, bgr0, bgr])])
        # bgr = np.vstack([np.hstack([frame3]), np.hstack([bgr])])
        cv2.imshow('frame', frame3)
        # cv2.imshow('gray', next)
        # cv2.imshow('frame2', frame2)

        k = cv2.waitKey(30) & 0xff
        # out.write(frame3)
        # out_2.write(frame2)
        if k == 27:  # K=ESC
            break
        # if count % 2 == 0:
        #     cv2.imwrite(r'flow_init' + str(count) + '.png', frame2)
        #     cv2.imwrite(r'flow' + str(count) + '.png', frame3)
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame3)
        #     # cv2.imwrite('opticalhsv.png', bgr)
        prvs = next  #


        if k == ord("r"):  # to make a reset of the flow
            # f0_2 = next_2
            f0 = next

        # if k == ord("w"):
        #     print("flow_t= ", flow_t)

    cap.release()
    cv2.destroyAllWindows()