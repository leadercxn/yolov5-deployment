'''
names = ["Alice","Bob","Carl"]
for index,value in enumerate(names):
    print(f'{index}: {value}')
'''

import cv2

#cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=640, height=480,type=video ! jpegdec ! videoconvert ! video/x-raw ! appsink")     # 读取视频
#cap = cv2.VideoCapture("/home/seeed/test/gd.jpg")
#cap = cv2.VideoCapture("/home/seeed/test/test.mp4")
cap = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",cv2.CAP_GSTREAMER)     # 读取视频
while cap.isOpened():               # 当视频被打开时：
    ret, frame = cap.read()         # 读取视频，读取到的某一帧存储到frame，若是读取成功，ret为True，反之为False
    if ret:                         # 若是读取成功
        cv2.imshow('frame', frame)  # 显示读取到的这一帧画面
        key = cv2.waitKey(25)       # 等待一段时间，并且检测键盘输入
        if key == ord('q'):         # 若是键盘输入'q',则退出，释放视频
            cap.release()           # 释放视频
            break
#    else:
#        cap.release()

cv2.destroyAllWindows()             # 关闭所有窗口


