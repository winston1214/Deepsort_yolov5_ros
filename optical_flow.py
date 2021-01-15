import cv2
import numpy as np
import math  # for cosine, sine calculation _hyeonuk
import matplotlib.pyplot as plt
from collections import Counter


def optical_flow(x, before, img):  # Lucas Kanade
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # start, end
    final_frame_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(30, 30), maxLevel=100,
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.05))
    feature_params = dict(maxCorners=10000,
                          qualityLevel=0.05,
                          minDistance=10,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(final_frame_gray, **feature_params)
    idx = 0
    for i in p0:  # range out delete
        if (i[0][0] <= c1[0]) or (i[0][0] >= c2[0]) or (i[0][1] <= c1[1]) or (i[0][1] >= c2[1]):
            p0 = np.delete(p0, idx, axis=0)
        else:
            idx += 1

    final_frame_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(final_frame_gray, final_frame_gray2, p0, None, **lk_params)
        # good_new = p1[st==1]
        # good_old = p0[st==1]
        # angle_array = []
        # mag_array = []
        mask = np.zeros_like(before)

        for f2, f1 in zip(p1, p0):
            a, b = f2.ravel()
            c, d = f1.ravel()
            # cv2.arrowedLine(img, (a, b), (c, d), (0, 0, 255), 2)
            cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
        return mask
    except:
        pass



def dense_optical_flow(x, before, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # start, end
    hsv = np.zeros_like(before, dtype=np.float64)
    output = np.zeros_like(before, dtype=np.float64)

    prvs = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

    hsv[..., 1], output[..., 1] = 255, 255
    # img = img[c1[1]:c2[1],c1[0]:c2[0]] # ROI만 계산
    next_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, pyr_scale=0.5, levels=6, winsize=15, iterations=3, poly_n=5, \
                                        poly_sigma=1.1,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    grid_x,grid_y = drawFlow(img,flow)
    motion = ego_motion(grid_x,grid_y,3)
    w,h = img.shape[:2]
    n=3
    count = 0
    for i in range(1,n):
        for j in range(1,n):
            cross_point[count] = (int(w // n * i), int(h // n * j))
            count += 1


    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # ang = radian

    hsv[..., 0] = ang  # *180/np.pi/2 # radian type
    hsv[..., 2] = mag

    output[c1[1]:c2[1], c1[0]:c2[0]] = hsv[c1[1]:c2[1], c1[0]:c2[0]]


    bbox = output[c1[1]:c2[1], c1[0]:c2[0]]

    angle_ls = bbox[..., 0]
    mag_ls = bbox[..., 2]


    angle_ls = angle_ls.flatten()
    mag_ls = mag_ls.flatten()


    x_ls = [a * math.cos(b) for a, b in zip(mag_ls, angle_ls)]
    y_ls = [a * math.sin(b) for a, b in zip(mag_ls, angle_ls)]

    x_mean = np.mean(x_ls)
    y_mean = np.mean(y_ls)

    return x_mean, y_mean  # return 값을 magnitude, angle이 아닌 x, y형태로 받음. _hyeonuk

def drawFlow(img,flow,step = 32):
    h,w = img.shape[:2]
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int)
    indices = np.stack((idx_x,idx_y),axis=-1).reshape(-1,2)
    grid_x = []
    grid_y = []

    for x,y in indices:
        #cv2.circle(img,(x,y),3,(0,255,0),-1)
        dx,dy = flow[y,x].astype(np.int) # 벡터
        grid_x.append(dx)
        grid_y.append(dy)


    grid_x = np.array(grid_x).reshape(-1,int(np.round(w/step)))
    grid_y = np.array(grid_y).reshape(-1,int(np.round(w/step)))
    return grid_x,grid_y

def ego_motion(grid_x,grid_y,n=3): # ego motion 추출
    w,h = grid_x.shape[:2] # 실제론 반대. 나중에 for문으로 돌릴 때 고쳐

    motion = []
    ego_motion1 = (np.mean(grid_x[:int(w//3),:int(h//3)]),np.mean(grid_y[:int(w//3),:int(h//3)]))
    ego_motion2 = (np.mean(grid_x[int(w//3):int(w//3*2),:int(h//3)]),np.mean(grid_y[int(w//3):int(w//3*2),:int(h//3)]))
    ego_motion3 = (np.mean(grid_x[int(w//3*2):,:int(h//3)]),np.mean(grid_y[int(w//3*2):,:int(h//3)]))
    ego_motion4 = (np.mean(grid_x[:int(w//3),int(h//3):int(h//3*2)]),np.mean(grid_y[:int(w//3),int(h//3):int(h//3*2)]))
    ego_motion5 = (np.mean(grid_x[int(w//3):int(w//3*2), int(h // 3):int(h // 3 * 2)]),np.mean(grid_y[int(w//3):int(w//3*2), int(h // 3):int(h // 3 * 2)]))
    ego_motion6 = (np.mean(grid_x[int(w//3*2):, int(h // 3):int(h // 3 * 2)]),np.mean(grid_y[int(w//3*2):, int(h // 3):int(h // 3 * 2)]))
    ego_motion7 = (np.mean(grid_x[:int(w//3),int(h//3*2):]),np.mean(grid_y[:int(w//3),int(h//3*2):]))
    ego_motion8 = (np.mean(grid_x[int(w//3):int(w//3*2), int(h // 3 * 2):]),np.mean(grid_y[int(w//3):int(w//3*2), int(h // 3 * 2):]))
    ego_motion9 = (np.mean(grid_x[int(w//3*2):, int(h // 3 * 2):]),np.mean(grid_y[int(w//3*2):, int(h // 3 * 2):]))
    for i in [ego_motion1,ego_motion2,ego_motion3,ego_motion4,ego_motion5,ego_motion6,ego_motion7,ego_motion8,ego_motion9]:
        motion.append(i)

    # for i in range(1,n+1):
    #     for j in range(1,n+1):
    #         grid_x_m = np.mean(grid_x[int(w//n*(i-1)):int(w//n*(i)),int(h//n*(j-1)):int(h//n*j)])
    #         grid_y_m = np.mean(grid_y[int(w//n*(i-1)):int(w//n*(i)),int(h//n*(j-1)):int(h//n*j)])
    #         motion.append((grid_x_m,grid_y_m))

    return motion