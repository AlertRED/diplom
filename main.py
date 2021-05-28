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
    def __init__(self, weights, img_size, conf, font_scale=0.5, iou_threshold=0.2):
        self.font_scale = font_scale
        self.weights = weights
        self.img_size = img_size
        self.conf = conf
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)  # custom model
        self.iou_threshold = iou_threshold
        self.mot_tracker = Sort(max_age=5, iou_threshold=self.iou_threshold, min_hits=3)

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

        # t = time.time()
        pred = self.model(img)[0]
        # print(time.time() - t)

        pred = non_max_suppression(pred, self.conf, self.iou_threshold, classes=None,
                                   agnostic=True)[0]
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

        bbox_thick = int(0.4 * (image_h + image_w) / self.img_size)

        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)

                # score = bbox[4]
                class_ind = int(bbox[5])
                # class_ind = class_ind if class_ind != 4 else 1
                # class_ind = 1
                bbox_color = colors[class_ind]

                c1, c2 = (int(coor[0] * w), int(coor[1] * h)), (int(coor[2] * w), int(coor[3] * h))
                rect_and_class.append([c1[0], c1[1], c2[0], c2[1], class_ind, i])

            nd = np.array(rect_and_class)
            track_bbs_ids = self.mot_tracker.update(nd)
            for obj in track_bbs_ids:
                x1, y1, x2, y2 = obj[:4].astype(int)
                _id = obj[4].astype(int)
                index = obj[5].astype(int)
                # name = classes[int(obj[6])]
                class_ind = int(bboxes[index][5])
                class_ind = class_ind if class_ind != 4 else 1

                m1 = '%d' % (_id)
                m2 = '%.2f' % (bboxes[index][4])
                m3 = '%s' % (classes[class_ind])
                cv2.rectangle(image, (x1, y1 - 10), (x2, y2 - 10), color=colors[class_ind])

                # cv2.rectangle(image, (x1, y1-10), (x1+20, y1+15), color=(255, 255, 255), thickness=-1)

                # x, y, w, h = 100, 100, 200, 100
                sub_img = image[y1 - 10:y2 - 10, x1:x2]
                if sub_img.size:
                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 0.1)

                    # Putting the image back to its position
                    image[y1 - 10:y2 - 10, x1:x2] = res

                cv2.putText(image, m1, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 255),
                            bbox_thick, lineType=cv2.LINE_AA)
                cv2.putText(image, m2, (x1, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), bbox_thick,
                            lineType=cv2.LINE_AA)
                cv2.putText(image, m3, (x1, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), bbox_thick,
                            lineType=cv2.LINE_AA)

        return image


if __name__ == '__main__':

    logging.disable()
    detect = Detect('resources/best_416_200.pt', 416, 0.4)

    cap = cv2.VideoCapture('resources/1.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

    prev = 0

    while (cap.isOpened()):

        time_elapsed = time.time() - prev
        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
            break

        if time_elapsed > 1. / 60:
            prev = time.time()

            frame = detect.detect_frame(frame)

            fps_str = 'FPS {0:.2}'.format(1 / (time.time() - prev))
            cv2.putText(frame, fps_str, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_4)
            cv2.imshow('frame', frame)
            out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
