import colorsys
import random
import time
import logging

import cv2
import numpy as np
import torch

from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

from sort.sort import *


class Detect():
    def __init__(self, weights, img_size, conf):
        self.weights = weights
        self.img_size = img_size
        self.conf = conf
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)  # custom model
        self.mot_tracker = Sort(max_age=5, iou_threshold=0.1, min_hits=2)

    def detect_frame(self, img0s):
        device = select_device('cpu')

        img = letterbox(img0s, self.img_size, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf, 0.45, classes=None,
                                   agnostic=False)[0]
        h, w = len(img[0][0]), len(img[0][0][0])
        return self.draw_bbox(img0s, pred, ['bicycle', 'car', 'motorcycle', 'bus', 'car'], w, h)


    def draw_bbox(self, image, bboxes, classes, w, h, show_label=True):
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        h = image_h / h
        w = image_w / w

        rect_and_class = []

        fontScale = 0.5
        bbox_thick = int(0.4 * (image_h + image_w) / self.img_size)

        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)

                score = bbox[4]
                class_ind = int(bbox[5])
                bbox_color = colors[class_ind]

                c1, c2 = (int(coor[0] * w), int(coor[1] * h)), (int(coor[2] * w), int(coor[3] * h))

                # cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

                rect_and_class.append([c1[0], c1[1], c2[0], c2[1], class_ind])

            nd = np.array(rect_and_class)
            track_bbs_ids = self.mot_tracker.update(nd)

            for obj in track_bbs_ids:
                x1, y1, x2, y2 = obj[:4].astype(int)
                _id = obj[4].astype(int)
                bbox_mess = '%d' % (_id)
                # t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), color=colors[0])

                cv2.putText(image, bbox_mess, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            print([i for i in track_bbs_ids[:,4]])


        return image


if __name__ == '__main__':
    logging.disable()

    detect = Detect('resources/best_0.pt', 480, 0.4)

    cap = cv2.VideoCapture('resources/1.mp4')
    prev = 0

    while (cap.isOpened()):
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if time_elapsed > 1. / 60:
            prev = time.time()
            frame = detect.detect_frame(frame)
            fps_str = 'FPS {0:.2}'.format(1 / (time.time() - prev))
            cv2.putText(frame, fps_str, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_4)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
