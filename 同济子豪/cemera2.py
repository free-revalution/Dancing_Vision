# opencv-python
import matplotlib
matplotlib.use('Agg')
# 运行程序前在终端写入，打开串口
# sudo chmod 777 /dev/ttyTHS1
import cv2
# mediapipe人工智能工具包
import mediapipe as mp
# 时间库
import time
# 串口通讯需要调用的库
import struct
import serial

# 定义串口
# com = serial.Serial("/dev/ttyTHS1", 115200)
# 定义变量
xpoint_07 = xpoint_08 = xpoint_11 = xpoint_12 = 0
ypoint_01 = ypoint_02 = ypoint_05 = ypoint_06 = 0
w = h = 0

BUF = bytearray(5)
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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 打开cap
cap.open(0)

# 串口通讯函数
def packData(number, b):
    print(BUF)
    struct.pack_into('<2B2b', BUF, 0xEF, 0, number, b, 0xEE)
    checksum = 0
    for i in BUF[:-1]:
        checksum += i
        checksum = (checksum & 0xff)
    struct.pack_into('<B', BUF, 4, checksum)
    # com.write(BUF)

def process_frame(img):

    global xpoint_07, xpoint_08, xpoint_11, xpoint_12 
    global ypoint_01, ypoint_02, ypoint_05, ypoint_06
    global w, h

    scaler = 1

    # 记录该帧开始处理的时间
    start_time = time.time()

    height = img.shape[0]
    width = img.shape[2]
    if width > 800:
        w = 800
        h = int(w/width*height)

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)


    if results.pose_landmarks:
        # 可视化
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # xpoint_01 = int(results.pose_landmarks.landmark[11].x * w)
        xpoint_02 = int(results.pose_landmarks.landmark[12].x * w)
        xpoint_03 = int(results.pose_landmarks.landmark[13].x * w)
        xpoint_04 = int(results.pose_landmarks.landmark[14].x * w)
        xpoint_05 = int(results.pose_landmarks.landmark[15].x * w)
        xpoint_06 = int(results.pose_landmarks.landmark[16].x * w)
        xpoint_07 = int(results.pose_landmarks.landmark[23].x * w)
        xpoint_08 = int(results.pose_landmarks.landmark[24].x * w)
        xpoint_09 = int(results.pose_landmarks.landmark[25].x * w)
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

        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1 / (end_time - start_time)
        '''
        0.未检测到人
        1.大字站
        2.弓箭步
        3.举双手
        4.蹲下
        5.自选动作一
        6.自选动作二
        7.检测到人但未检测出相应动作

        '''
        if (ypoint_06 - ypoint_02 > 0 and ypoint_05 - ypoint_01 > 0 and ypoint_06 - ypoint_05 > 0):
            # 大字站
            number = 1
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_01 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        elif (ypoint_06 - ypoint_02 > 0 and ypoint_05 - ypoint_01 < 0):
            # 弓箭步
            number = 2
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_02 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        elif (ypoint_06 - ypoint_02 < 0 and ypoint_05 - ypoint_01 < 0 and ypoint_06 - ypoint_05 < 0):
            # 举双手
            number = 3
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_03 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        elif (xpoint_06 - xpoint_05 > 0 and xpoint_09 - xpoint_11 > 0):
            # 蹲下
            number = 4
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_04 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        elif (ypoint_06 - ypoint_02 > 0 and ypoint_05 - ypoint_01 > 0 and ypoint_06 - ypoint_05 < 0):
            # 自选动作一，左侧平举
            number = 5
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_05 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        elif (ypoint_06 - ypoint_02 < 0 and ypoint_05 - ypoint_01 > 0):
            # 自选动作二，抱拳
            number = 6
            # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            img = cv2.putText(img, 'action_06 ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        else:
            number = 7
            failure_str = 'No useful information'
            img = cv2.putText(img, failure_str , (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        
    else:
        # 检测到人但未检测出相应动作
        number = 0
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        print('从图像中未检测出人体关键点，报错。')

    return img, number

while cap.isOpened():

    number = 0
    b = 0
    # 获取画面
    success, frame = cap.read()
    # 处理帧函数
    frame, number = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    print(number)
    # packData(int(number), b)

    # 按键盘上的q或esc退出（在英文输入法下）
    if cv2.waitKey(1) in [ord('q'), 27]:
        break

# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()