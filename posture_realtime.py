import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model

tic = 0
# visualize 颜色随机
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process(input_image, params, model_params):
    """ Start of finding the Key points of full body using Open Pose."""
    oriImg = input_image  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    # 计算乘子，比例缩放box的大小
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))      # 返回一个全0填充的三维矩阵heatmap
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))          # 返回一个全0填充的三位矩阵paf
    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # scale---0.5\1 * 368 / height
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])     # 右下角填充灰色，使宽、高像素是8的倍数
        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)     # output的宽高都缩小了8倍，这里恢复到与input_img相同
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]    # remove padding
        # resize到oriImg
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)   # scale的结果求平均

    all_peaks = []  # To store all the key points which a re detected.
    peak_counter = 0

    prinfTick(1)  # prints time required till now.

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)     # 高斯去噪
        # 找到峰值（当前像素值大小比上下左右的都大）
        map_left = np.zeros(map.shape)  # 0矩阵
        map_left[1:, :] = map[:-1, :]   # 取map左边的右移，map_left左边0
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        # 这里输出：像素值都是T or F，峰值T，图像大小和原图一样，最大值为T
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        # 输出T的坐标，即是峰值的一系列坐标 [(h1, w1), (h2, w2), (h3, w3), (h4, w4)]，此处坐标与原图是反转的(x,y反转了)
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        # 输出坐标即score，原图像素值作为score。 [(h1, w1, s1), (h2, w2, s2), (h3, w3 ,s3), (h4, w4, s4)]
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        ID = range(peak_counter, peak_counter + len(peaks))     # peaks长度
        peaks_with_score_and_id = [peaks_with_score[i] + (ID[i],) for i in range(len(ID))]

        all_peaks.append(peaks_with_score_and_id)   # 所有part的峰值全存入
        # all_peaks=[ [((h0, w0, s0,0),(h1, w1, s1,1)....]\  第一个part的所有值
        #            [((hi, wi, si,i),(hi+1, wi+1, si+1,i+1)....]\
        #               .....
        #           ]

        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    prinfTick(2)  # prints time required till now.

    canvas = frame  # B,G,R order

    for i in range(18):  # drawing all the detected key points.
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            # 画圆（图片，坐标，半径，颜色，-1：填充）
    print()
    flag = check(all_peaks)
    print(flag)
    if flag:
        checkNeck(all_peaks)
        checkShoulder(all_peaks)
        return canvas, 0
    else:
        position = checkPosition(all_peaks)
        checkKneeling(all_peaks)
        checkHandFold(all_peaks)
        checkNeck2(all_peaks)
        return canvas, position

    # print()
    # print()


def check(all_peaks):
    try:
        f = 0
        if all_peaks[2][0][0:2]:
            try:
                if all_peaks[5][0][0:2]:
                    distance = calcDistance(all_peaks[2][0][0:2],
                                            all_peaks[5][0][0:2])  # Distance between left shoulder and right shoulder
                    if distance > 200:
                        f = 1
                        return f
                    else:
                        f = 0
                        return f
            except Exception as e:
                print("没发现肩膀")
        return f
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")


def checkShoulder(all_peaks):
    try:
        c = all_peaks[2][0][0:2]  # right shoulder
        d = all_peaks[5][0][0:2]  # left shoulder
        angle1 = calcAngle(c, d)
        degrees1 = round(math.degrees(angle1))  # 弧度转角度：肩膀的角度，并且四舍五入得到整数
        if degrees1 > 6:
            print("Shoulder_left")
        elif degrees1 < -6:
            print("Shoulder_right")
        else:
            print("Shoulder_level")
    except Exception as e:
        print("error")


def checkNeck(all_peaks):
    try:
        a = all_peaks[0][0][0:2]  # nose
        b = all_peaks[1][0][0:2]  # neck
        angle = calcAngle(a, b)
        degrees = round(math.degrees(angle))  # 弧度转角度：脖子的角度，并且四舍五入得到整数
        if degrees > 100:   # 脖子左倾,偏离10°
            print("Neck_left")
        elif degrees < 80:   # 脖子右倾,偏离10°
            print("Neck_right")
        else:
            print("Neck_straight")
    except Exception as e:
        print("error")


def checkNeck2(all_peaks):
    try:
        a = all_peaks[0][0][0:2]    # Nose
        b = all_peaks[1][0][0:2]    # Neck
        angle = calcAngle(a, b)
        degrees = round(math.degrees(angle))    # 弧度转角度：脖子的角度，并且四舍五入得到整数
        # print(degrees)
        if degrees < 40:
            print("Neck forward")
        elif degrees > 140:
            print("Neck forward")
        else:
            print("Neck straight")
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")


def checkPosition(all_peaks):
    try:
        f = 0
        if all_peaks[16]:
            a = all_peaks[16][0][0:2]  # Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2]  # Left Ear
        b = all_peaks[11][0][0:2]  # Hip
        c = all_peaks[0][0][0:2]   # Nose
        d = all_peaks[1][0][0:2]   # Neck
        angle = calcAngle(a, b)
        angle1 = calcAngle(c, d)
        degrees = round(math.degrees(angle))    # 弧度转角度：耳朵与臀部的角度，并且四舍五入得到整数
        degrees1 = round(math.degrees(angle1))  # 弧度转角度：脖子的角度，并且四舍五入得到整数
        # print(degrees1)
        if f:
            degrees = 180 - degrees
        if degrees < 70:
            position =  1    # 前倾
        elif degrees > 110:
            position = -1
        else:
            position = 0
        return position
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")


def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if ax == bx:
            return 1.570796     # 角度：90
        return math.atan2(by - ay, bx - ax)
    except Exception as e:
        print("unable to calculate angle")


def checkHandFold(all_peaks):
    try:
        if all_peaks[3][0][0:2]:
            try:
                if all_peaks[4][0][0:2]:
                    distance = calcDistance(all_peaks[3][0][0:2],
                                            all_peaks[4][0][0:2])  # distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2],
                                           all_peaks[3][0][0:2])  # distance between left arm-joint and left palm.
                    if ((armdist + 100) > distance > (
                            armdist - 100)):
                        # this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
                        print("Not Folding Hands")
                    else:
                        print("Folding Hands")
            except Exception as e:
                print("Folding Hands")
    except Exception as e:
        try:
            if all_peaks[7][0][0:2]:
                distance = calcDistance(all_peaks[6][0][0:2], all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if (armdist + 100) > distance > (armdist - 100):
                    print("Not Folding Hands")
                else:
                    print("Folding Hands")
        except Exception as e:
            print("Unable to detect arm joints")


def calcDistance(a, b):  # calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")


def checkKneeling(all_peaks):
    f = 0
    if all_peaks[16]:
        f = 1
    try:
        if all_peaks[10][0][0:2] and all_peaks[13][0][0:2]:
            rightankle = all_peaks[10][0][0:2]
            leftankle = all_peaks[13][0][0:2]
            hip = all_peaks[11][0][0:2]
            leftangle = calcAngle(hip, leftankle)
            leftdegrees = round(math.degrees(leftangle))
            rightangle = calcAngle(hip, rightankle)
            rightdegrees = round(math.degrees(rightangle))
        if (f == 0):
            leftdegrees = 180 - leftdegrees
            rightdegrees = 180 - rightdegrees
        if (
                leftdegrees > 60 and rightdegrees > 60):
            # 60 degrees is trail and error value here. We can tweak this accordingly and results will vary.
            print("Both Legs are in Kneeling")
        elif rightdegrees > 60:
            print("Right leg is kneeling")
        elif leftdegrees > 60:
            print("Left leg is kneeling")
        else:
            print("Not kneeling")

    except IndexError as e:
        try:
            if f:
                a = all_peaks[10][0][0:2]  # if only one leg (right leg) is detected
            else:
                a = all_peaks[13][0][0:2]  # if only one leg (left leg) is detected
            b = all_peaks[11][0][0:2]  # location of hip
            angle = calcAngle(b, a)
            degrees = round(math.degrees(angle))
            if f == 0:
                degrees = 180 - degrees
            if degrees > 60:
                print("Both Legs Kneeling")
            else:
                print("Not Kneeling")
        except Exception as e:
            print("legs not detected")


def prinfTick(i):
    toc = time.time()
    print('processing time%d is %.5f' % (i, toc - tic))


if __name__ == '__main__':

    tic = time.time()
    print('start processing...')
    model = get_testing_model()
    model.load_weights('./model/keras/model.h5')

    cap = cv2.VideoCapture(0)   # 0---打开本机摄像头
    vi = cap.isOpened()         # ture---已打开

    if vi:
        cap.set(100, 160)
        cap.set(200, 120)
        time.sleep(2)

        while 1:
            tic = time.time()   # 返回当前时间
            ret, frame = cap.read()     # ret---ture成功read，frame---一帧图片
            params, model_params = config_reader()      # 返回config中配置的参数
            canvas, position = process(frame, params, model_params)
            if position == 1:
                print("Hunchback")
            elif position == -1:
                print("Reclined")
            elif position == 0:
                print("Straight")
            print()
            cv2.imshow("capture", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        print("unable to open camera")
cv2.destroyAllWindows()
