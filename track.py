import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
from optical_flow import dense_optical_flow
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



img_ls = []
vector_qx = []
vector_qy = []
def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    rail_pts = np.array([[988, 357], [1107, 357], [1643, 1073], [577, 1073]], np.int32)
    slope1 = (rail_pts[0][1]-rail_pts[-1][1])/(rail_pts[0][0]-rail_pts[-1][0])
    slope2 = (rail_pts[2][1]-rail_pts[1][1])/(rail_pts[2][0]-rail_pts[1][0])
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            #im0 = cv2.resize(im0,dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            cv2.polylines(im0, [rail_pts], True, (255, 0, 255), 3)
            img_ls.append(im0)

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                annotation = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    annotation.append(names[int(c)])


                    if len(bbox_xywh) == 1:
                        if (x2 <= im0.shape[1] // 2):
                            location = 'left'
                        elif (x1 >= im0.shape[1] // 2):
                            location = 'right'
                        elif (x2 >= im0.shape[1] // 2 and x1 <= im0.shape[1] // 2):
                            location = 'center'
                    if (y2 <= -slope1*(x2 - rail_pts[0][0]) + rail_pts[0][1]) and location == 'left':
                        cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
                        break
                    elif (y2 <= -slope2*(x1 - rail_pts[2][0])+rail_pts[2][1]) and location == 'right':
                        cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
                        break

                    if len(img_ls) >= 2:  # changed 10 to 2 _hyeonuk
                        before = img_ls[-2]
                        cur = img_ls[-1]
                        if len(img_ls) >=3:
                            img_ls.pop(0)

                        mo_x, mo_y = dense_optical_flow(xyxy, before, cur)

                        if len(vector_qx) > 5:
                            vector_qx.pop(0)
                            vector_qy.pop(0)
                            vector_qx.append(mo_x)
                            vector_qy.append(mo_y)
                            x_sum = sum(vector_qx)
                            y_sum = sum(vector_qy)
                            x_mean = x_sum  # / len(vector_qx)
                            y_mean = y_sum  # / len(vector_qy)
                        else:
                            vector_qx.append(mo_x)
                            vector_qy.append(mo_y)
                            x_sum = sum(vector_qx)
                            y_sum = sum(vector_qy)
                            x_mean = x_sum  # / len(vector_qx)
                            y_mean = y_sum  # / len(vector_qy)
                            # 대표백터 magnitude angle로 변경 _hyeonuk
                        mag = np.sqrt((math.pow(x_mean, 2)) + (math.pow(y_mean, 2)))  # 수정 예
                        degree = np.angle(complex(x_mean,y_mean),deg=True)


                        if location == -1:  # (x2<=im0.shape[1]//2): # left
                            if (degree > 270) and (degree <= 315):  # magnitude 조건 추가 예정
                                cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),
                                            3)

                            else:
                                cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)
                        elif location == 1:  # (x1>=im0.shape[1]//2) : # right
                            if (degree <= 270) and (degree > 225):
                                cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),
                                            3)


                            else:
                                cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)
                        elif location == 0:  # (x2 >= im0.shape[1]//2 and x1<=im0.shape[1]//2): # center
                            if (np.mean([225, 270]) <= degree) and degree <= np.mean([270, 315]):
                                cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),
                                            3)

                            else:
                                cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        new_x = center_x + mag * np.cos(math.radians(degree))
                        new_y = center_y + mag * np.sin(math.radians(degree))
                        cv2.arrowedLine(im0, (int(center_x), int(center_y)), (int(new_x), int(new_y)), (0, 0, 255), 5)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        # im0 = cv2.resize(im0, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person , class 1 is car
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
