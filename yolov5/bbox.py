import cv2
import numpy as np
import math
import time
from vector import calc_vector

frame_cnt = int(input("Enter the number of frame: "))
input_data = np.array([[174,139,266,453],[151,137,264,462],[127,139,237,472],[117,143,231,480],[98,141,262,496],
                      [80,153,256,494],[80,143,253,515],[127,141,258,531],[119,151,284,529],[119,156,268,549],
                       [151,151,245,536],[155,153,280,551],[151,162,301,566],[149,159,292,580], [157,160,280,597],
                       [147,166,284,615],[145,174,301,636],[150,165,282,641],[161,163,290,672], [152,178,334,681],
                       [143,184,336,722],[142,174,312,737], [58,182,211,848]]) # bounding box 좌표

vector_result, P1, P2, center, next_center, next_P2, camera_distance = calc_vector(input_data, 6)
rail_pts = np.array([[732, 291], [943, 291], [1284, 1073], [100, 1073]], np.int32) # rail의 좌표
empty_img = np.zeros((1086, 2040, 3), np.uint8)

for i in range(frame_cnt):
    road_image = cv2.imread("./image/%d.jpg" % i)
    clone = road_image.copy()

    cv2.polylines(clone, [rail_pts], True, (255, 0, 255), 3)

    # 바운딩 박스 조건문
    if i >= 21:
        if (((rail_pts[0][0] + rail_pts[3][0]) / 2) + rail_pts[3][0]) / 2 - next_P2[i][0] >= 0:
            cv2.rectangle(clone, (P1[i][0], P1[i][1]), (P2[i][0], P2[i][1]), (0, 0, 255), 2)
            cv2.putText(clone, "STATUS : DANGER", (1020, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.rectangle(clone, (P1[i][0], P1[i][1]), (P2[i][0], P2[i][1]), (0, 255, 0), 2)
    else:
        cv2.rectangle(clone, (P1[i][0], P1[i][1]), (P2[i][0], P2[i][1]), (0, 255, 0), 2)
        cv2.putText(clone, "STATUS : SAFE", (1020, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 125, 255), 3)

    tram_velocity = 2.78 - (0.1418 * (i + 1) * 0.09591)
    TTC = camera_distance[i] / tram_velocity
    print("%d frame:" % i, "distance -> %f" % camera_distance[i], "velocity -> %f" % tram_velocity, "TTC -> %.2fs" % TTC)
    cv2.putText(clone, "TTC : %.2fs" % TTC, (130, 130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 3)

    if i > 0:
        arrow_px = int(center[i][0]) + 50 * math.cos(math.radians(vector_result[i-1][1]))
        arrow_py = int(center[i][1]) + 50 * math.sin(math.radians(vector_result[i-1][1]))
        cv2.arrowedLine(clone, (int(center[i][0]), int(center[i][1])), (int(arrow_px), int(arrow_py)),
                        (0, 0, 255), 12, cv2.LINE_AA)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.1)
    clone = cv2.imread("./image/result/temp%d.jpg" % i)
