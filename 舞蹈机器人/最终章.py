# opencv-python
import matplotlib
matplotlib.use('Agg')

# 运行程序前在终端写入，打开串口
# sudo chmod 777 /dev/ttyAMA0

import cv2
# mediapipe人工智能工具包
import mediapipe as mp
# 时间库
import time
# 串口通讯需要调用的库
import struct
import serial
import numpy as np
# 多线程函数
from threading import Thread
import queue
import threading
import math

# 定义串口
# ser = serial.Serial('/dev/ttyAMA0',115200)
# 串口初始化，根据实际情况修改串口号和波特率

# 定义变量
xpoint_01 = xpoint_02 = xpoint_05 = xpoint_06 = 0
ypoint_01 = ypoint_02 = ypoint_05 = ypoint_06 = 0
w = h = 0
x_ray = y_ray = []

# 线程计数变量
exitFlag = 0

# 导入solution
mp_pose = mp.solutions.pose
# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,  # 是静态图片还是连续视频帧
                    model_complexity=1,  # 选择人体姿态关键点检测模型，0 性能差但快，2 性能好但慢，1介于两者之间
                    smooth_landmarks=True,  # 是否平滑关键点
                    enable_segmentation=True,  # 是否人体抠图
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)  # 追踪阈值

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 打开cap
cap.open(0)

def num_choose(num):

    count = 0
    while(count < 21):

        lisi_nums = []
        lisi_nums.append(num)
        count = count + 1

    max_times = max(lisi_nums, key=lisi_nums.count)

    return max_times


def process_frame(img):

    global xpoint_01, xpoint_02, xpoint_05, xpoint_06
    global ypoint_01, ypoint_02, ypoint_05, ypoint_06
    global w, h
    global x_ray, y_ray

    scaler = 1
    # 记录该帧开始处理的时间
    start_time = time.time()

    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)



    if results.pose_landmarks:
        # 可视化
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 得出所有关键点的坐标
        xpoint_01 = int(results.pose_landmarks.landmark[11].x * w)
        xpoint_02 = int(results.pose_landmarks.landmark[12].x * w)
        xpoint_03 = int(results.pose_landmarks.landmark[13].x * w)
        xpoint_04 = int(results.pose_landmarks.landmark[14].x * w)
        xpoint_05 = int(results.pose_landmarks.landmark[15].x * w)
        xpoint_06 = int(results.pose_landmarks.landmark[16].x * w)
        xpoint_07 = int(results.pose_landmarks.landmark[23].x * w)
        xpoint_08 = int(results.pose_landmarks.landmark[24].x * w)
        xpoint_09 = int(results.pose_landmarks.landmark[25].x * w)
        xpoint_10 = int(results.pose_landmarks.landmark[26].x * h)
        xpoint_11 = int(results.pose_landmarks.landmark[27].x * w)
        xpoint_12 = int(results.pose_landmarks.landmark[28].x * w)

        ypoint_01 = int(results.pose_landmarks.landmark[11].y * h)
        ypoint_02 = int(results.pose_landmarks.landmark[12].y * h)
        ypoint_03 = int(results.pose_landmarks.landmark[13].y * h)
        ypoint_04 = int(results.pose_landmarks.landmark[14].y * h)
        ypoint_05 = int(results.pose_landmarks.landmark[15].y * h)
        ypoint_06 = int(results.pose_landmarks.landmark[16].y * h)
        ypoint_07 = int(results.pose_landmarks.landmark[23].y * h)
        ypoint_08 = int(results.pose_landmarks.landmark[24].y * h)
        ypoint_09 = int(results.pose_landmarks.landmark[25].y * h)
        ypoint_10 = int(results.pose_landmarks.landmark[26].y * h)
        ypoint_11 = int(results.pose_landmarks.landmark[27].y * h)
        ypoint_12 = int(results.pose_landmarks.landmark[28].y * h)

        x_ray = [xpoint_01, xpoint_02, xpoint_03, xpoint_04, xpoint_05, xpoint_06, xpoint_07, xpoint_08, xpoint_09,
                 xpoint_10, xpoint_11, xpoint_12]
        y_ray = [ypoint_01, ypoint_02, ypoint_03, ypoint_04, ypoint_05, ypoint_06, ypoint_07, ypoint_08, ypoint_09,
                 ypoint_10, ypoint_11, ypoint_12]

        # 如果没有数据输出，在此打印数据
        # print(xpoint_01, xpoint_02, xpoint_05, xpoint_06, ypoint_01, ypoint_02, ypoint_05, ypoint_06)

        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1 / (end_time - start_time)

        failure_str = 'Person'
        img = cv2.putText(img, failure_str + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)
    else:
        # 未检测到人而且未检测出相应动作
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)


    return img, x_ray, y_ray


def get_angle(point_1, point_2, point_3):
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    # angle_A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    angle_B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    # angle_C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    return angle_B




while cap.isOpened():

    number = 0
    b = 0
    # 获取画面
    success, frame = cap.read()
    # 处理帧函数
    frame, x_ray, y_ray = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)
    # print(x_ray[0]) y_ray[0]
    # （中间点坐标，外侧点坐标，内侧点坐标）

    final_point_01 = [(x_ray[4], y_ray[4]), (x_ray[1], y_ray[1]), (x_ray[10], y_ray[10]), (x_ray[11], y_ray[11])]
    final_point_02 = [(x_ray[2], y_ray[2]), (x_ray[3], y_ray[3]), (x_ray[8], y_ray[8]), (x_ray[9], y_ray[9])]
    final_point_03 = [(x_ray[0], y_ray[0]), (x_ray[5], y_ray[5]), (x_ray[6], y_ray[6]), (x_ray[7], y_ray[7])]
    # 右膀臂角度
    final_angle_01 = int(get_angle(final_point_01[0], final_point_02[0], final_point_03[0]))
    # 左膀臂角度
    final_angle_02 = int(get_angle(final_point_01[1], final_point_02[1], final_point_03[1]))
    # 右腿角度
    final_angle_03 = int(get_angle(final_point_01[2], final_point_02[2], final_point_03[2]))
    # 左腿角度
    final_angle_04 = int(get_angle(final_point_01[3], final_point_02[3], final_point_03[3]))

    # final_num = num_choose(number)

    # print(final_num)
    print(final_angle_01, final_angle_02, final_angle_03, final_angle_04)

    # ser.write(str(int(final_num)).encode())  # 发送数字到串口

    # 按键盘上的q或esc退出（在英文输入法下）
    if cv2.waitKey(1) in [ord('q'), 27]:
        break

# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows