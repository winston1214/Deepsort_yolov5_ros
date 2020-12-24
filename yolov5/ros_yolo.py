#!/usr/bin/env python3

#@Author YounminKim(winston1214)
import rospy
import cv2
from cv_bridge import CvBridge,CvBridgeError
import numpy as np
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch.backends.cudnn as cudnn
from numpy import random
import math
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
from vector import calc_vector

class ROS:
    def __init__(self):
        self.image_topic = '/pylon_camera_node1/image_raw'
        self.fps = 30
        self.bridge = CvBridge()
        self.input_data = []
        self.image_sub = rospy.Subscriber(self.image_topic,Image,self.image_callback,queue_size=1)
        self.path = 'ros.jpg'
        self.rviz_pub = rospy.Publisher('yolov5_image',Image,queue_size=1)
        self.ttc_pub = rospy.Publisher('TTC',String,queue_size=1)
        
        rospy.spin()
    def image_callback(self,msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except CvBridgeError as e:
            print(e)

        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        ros_img = source == 'ros'
        rail_pts = np.array([[988,357], [1107, 357], [1643, 1073], [577, 1073]], np.int32)
        #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        set_logging()
        if ros_img:
            view_img = True
            cudnn.benchmark = True 
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


        im0s = self.cv_image # bgr
        img = letterbox(im0s, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device) # rgb
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for _, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(self.path), '', im0s
            cv2.polylines(im0, [rail_pts], True, (255, 0, 255), 3)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if view_img:  # Add bbox to image
                    label = '%s' % (names[int(cls)])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                    self.input_data.append([x1,y1,x2,y2])
                    idx = len(self.input_data)-1

                    if len(self.input_data) >= 2:
                        vector_result, P1, P2, center, next_center, next_P2, camera_distance = calc_vector(np.array(self.input_data), self.fps)
                        tram_velocity = 2.78 - (0.0773 * (idx + 1) * (1/30)) # 가감속속도
                        TTC = camera_distance[idx]/tram_velocity
                        

                        print("%d frame:" % idx, "distance -> %f" % camera_distance[idx], "velocity -> %f" % tram_velocity, "TTC -> %.2fs" % TTC)
                        # cv2.putText(im0, "TTC : %.2fs" % TTC, (130, 130), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 3)
                        arrow_px = int(center[idx][0]) + 50 * math.cos(math.radians(vector_result[idx-1][1]))
                        arrow_py = int(center[idx][1]) + 50 * math.sin(math.radians(vector_result[idx-1][1]))
                        cv2.arrowedLine(im0, (int(center[idx][0]), int(center[idx][1])), (int(arrow_px), int(arrow_py)),
                                        (0, 0, 255), 12, cv2.LINE_AA)
                        arrow_px2 = int(arrow_px) +  (P2[idx][1]-P1[idx][1]) * math.cos(math.radians(vector_result[idx-1][1]))
                        arrow_py2 = int(arrow_py) +  (P2[idx][1]-P1[idx][1]) * math.sin(math.radians(vector_result[idx-1][1]))
                        if (TTC <=5) and (arrow_px2>=rail_pts[3][0]) and (arrow_px2<=rail_pts[2][0]) and (camera_distance[idx]<=10):
                            cv2.putText(im0, "STATUS : DANGER", (1020, 50),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
                            status='danger'
                            cv2.circle(im0,(int(arrow_px2),int(arrow_py2)),color=(0,255,0),radius = 10,thickness=-1)
                            cv2.putText(im0,"Predicted Colision point",(int(arrow_px2-200),int(arrow_py2)+50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,204),3)
                        else:
                            cv2.putText(im0, "STATUS : SAFE", (1020, 50),cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 125, 255), 3)
                            status='safe'
                    

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(str(p), im0)
                image_msg = self.bridge.cv2_to_imgmsg(im0, "bgr8")
                self.rviz_pub.publish(image_msg)
                try:
                    self.ttc_pub.publish('TTC %.2f %s'%(TTC,status))
                except: pass
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()

        print('Done. (%.3fs)' % (time.time() - t0))
if __name__=="__main__":
    # Initialize node
    
    rospy.init_node("detector_manager_node",anonymous=True,disable_signals=True)
    # Define detector object
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        ROS()
