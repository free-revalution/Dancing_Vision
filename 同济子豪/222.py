import cv2
# mediapipe人工智能工具包
import mediapipe as mp
# 进度条库
from tqdm import tqdm
# 时间库
import time


# 导入solution
mp_pose = mp.solutions.pose

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=1,             # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,          # 是否平滑关键点
                    min_detection_confidence=0.5,   # 置信度阈值
                    min_tracking_confidence=0.5)    # 追踪阈值


def process_frame(img):

    # 记录该帧开始处理的时间
    start_time = time.time()

    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks:  # 若检测出人体关键点

        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):  # 遍历所有33个关键点，可视化

            # 获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z

            radius = 10

            if i == 0:  # 鼻尖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [11, 12]:  # 肩膀
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
            elif i in [23, 24]:  # 髋关节
                img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
            elif i in [13, 14]:  # 胳膊肘
                img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
            elif i in [25, 26]:  # 膝盖
                img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:  # 手腕和脚腕
                img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
            elif i in [17, 19, 21]:  # 左手
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            elif i in [18, 20, 22]:  # 右手
                img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
            elif i in [27, 29, 31]:  # 左脚
                img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
            elif i in [28, 30, 32]:  # 右脚
                img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
            elif i in [9, 10]:  # 嘴
                img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 眼及脸颊
                img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
            else:  # 其它关键点
                img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)

        # 展示图片
        # look_img(img)

    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)
        # print('从图像中未检测出人体关键点，报错。')

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                      (255, 0, 255), 2 * scaler)
    return img


# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        break

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()