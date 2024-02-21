# opencv-python
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

BUF = bytearray(5)


# 串口通讯函数
def packData(a, b):
    print(BUF)
    struct.pack_into('<2B2b', BUF, 0, 0x2C, 0x12, a, b, )
    checksum = 0
    for i in BUF[:-1]:
        checksum += i
        checksum = (checksum & 0xff)
    struct.pack_into('<B', BUF, 4, checksum)

    # com.write(BUF)


# 处理帧函数
def process_frame(img):

    global xpoint_07, xpoint_08, xpoint_11, xpoint_12
    global ypoint_01, ypoint_02, ypoint_05, ypoint_06
    # 记录该帧开始处理的时间
    start_time = time.time()

    h = img.shape[0]
    w = img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)
    if results.pose_landmarks:  # 若检测出人体关键点
        # 可视化
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        xpoint_07 = int(results.pose_landmarks.landmark[23].x * w)
        xpoint_08 = int(results.pose_landmarks.landmark[24].x * w)
        xpoint_11 = int(results.pose_landmarks.landmark[27].x * w)
        xpoint_12 = int(results.pose_landmarks.landmark[28].x * w)

        ypoint_01 = int(results.pose_landmarks.landmark[11].y * h)
        ypoint_02 = int(results.pose_landmarks.landmark[12].y * h)
        ypoint_05 = int(results.pose_landmarks.landmark[15].y * h)
        ypoint_06 = int(results.pose_landmarks.landmark[16].y * h)
    else:
        print('从图像中未检测出人体关键点，报错。')

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    scaler = 1

    if (ypoint_06 - ypoint_02 > 0 & ypoint_05 - ypoint_01 > 0):
        number = 1
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, '动作一  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    elif (ypoint_06 - ypoint_02 < 0 & ypoint_05 - ypoint_01 > 0):
        number = 2
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, '动作二  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    elif (ypoint_06 == ypoint_05 & xpoint_12 - xpoint_08 > 0 & xpoint_11 - xpoint_07 < 0):
        number = 3
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, '动作三  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    elif (ypoint_06 == ypoint_05 & xpoint_11 - xpoint_07 < 0):
        number = 4
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, '动作四  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    else:
        number = 5
        failure_str = 'No useful information'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)
    return img, number


# 无限循环，直到break被触发
while cap.isOpened():

    number = 0
    b = 0
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break
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
