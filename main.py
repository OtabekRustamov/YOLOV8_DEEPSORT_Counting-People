# Ultralytics YOLO 🚀, GPL-3.0 license

from collections import deque

import cv2
import torch
from ultralytics.yolo import utils
from ultralytics.yolo.v8 import detect

from nets import nn

data_deque = {}
decision_boundary = 50  # in pixels


def draw_boxes(image, boxes, object_id, identities, sum_in, sum_out):
    h, w, _ = image.shape
    # cv2.line(image, (10, int(1 / 4 * h)), (w - 10, int(1 / 4 * h)), (255, 0, 255), 3)
    cv2.line(image, (10, int(1/ 2 * h)), (w - 10, int(1 / 2 * h)), (255, 0, 255), 3)

    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(boxes):
        if object_id[i] != 0:
            continue
        x1, y1, x2, y2 = list(map(int, box))

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        index = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if index not in data_deque:
            data_deque[index] = deque(maxlen=64)

        # add center to buffer
        data_deque[index].appendleft(center)

        # decide the object in or out
        if len(data_deque[index]) >= 2:
            point1 = data_deque[index][0]
            point2 = data_deque[index][1]
            if point1[1] < point2[1] and point2[1] > 3 / 4 * h:
                sum_in[index] = 1
            if point1[1] > point2[1] and point2[1] < 1 / 4 * h:
                sum_out[index] = 1

        cv2.circle(image, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), 5, (0, 255, 0), -1)

        # Display count in top left corner
        cv2.line(image, (0, 25), (500, 25), [0, 0, 255], 40)
        cv2.putText(image,
                    f'Numbers of Leaving: {sum(sum_out.values())}',
                    (11, 35), 0, 1, [225, 255, 255],
                    thickness=2, lineType=cv2.LINE_AA)
        # Display count in top right corner
        cv2.line(image, (w - 500, 25), (w, 25), [0, 0, 255], 40)
        cv2.putText(image,
                    f'Number of Entering: {sum(sum_in.values())}',
                    (w - 500, 35), 0, 1, [225, 255, 255],
                    thickness=2, lineType=cv2.LINE_AA)
    return sum_in, sum_out


class Predictor(detect.DetectionPredictor):
    def __init__(self, cfg=utils.DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)

        self.sum_in = {}
        self.sum_out = {}
        self.deepsort = nn.DeepSort()

    def write_results(self, index, results, batch):
        self.seen += 1
        p, im, image = batch
        image = image.copy()

        self.data_path = p
        self.annotator = self.get_annotator(image)

        if len(results[index].boxes) == 0:
            return ""

        boxes = []
        confidences = []
        object_indices = []
        for box in reversed(results[index].boxes):
            boxes.append(box.xywh.view(-1).tolist())
            confidences.append([box.conf.squeeze().item()])
            object_indices.append(int(box.cls.squeeze().item()))

        outputs = self.deepsort.update(torch.Tensor(boxes),
                                       torch.Tensor(confidences),
                                       object_indices, image)
        if len(outputs) > 0:
            self.sum_in, self.sum_out = draw_boxes(image,
                                                   outputs[:, :4],
                                                   outputs[:, -1],
                                                   outputs[:, -2],
                                                   self.sum_in, self.sum_out)

        return ""


def main():
    utils.DEFAULT_CFG.show = False
    utils.DEFAULT_CFG.mode = "predict"
    utils.DEFAULT_CFG.model = "weights/yolov8n.pt"
    utils.DEFAULT_CFG.source = "https://youtu.be/rfkGy6dwWJs"

    Predictor(utils.DEFAULT_CFG)()


if __name__ == "__main__":
    main()
