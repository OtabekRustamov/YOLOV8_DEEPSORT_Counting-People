import cv2
import numpy
import torch
from torchvision import transforms

from utils import util


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=None, d=1, g=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, bias)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        identity = torch.nn.Identity()
        self.add = s != 1 or in_ch != out_ch
        self.relu = torch.nn.ReLU(inplace=False)

        self.conv1 = Conv(in_ch, out_ch, self.relu, 3, s, 1)
        self.conv2 = Conv(out_ch, out_ch, identity, 3, 1, 1)

        if self.add:
            self.conv3 = Conv(in_ch, out_ch, identity, 1, s)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add:
            x = self.conv3(x)

        return self.relu(x + y)


class ReID(torch.nn.Module):
    def __init__(self):
        super().__init__()
        filters = [3, 64, 128, 256, 512]
        p1 = [Conv(filters[0], filters[1], torch.nn.ReLU(), 3, bias=True),
              torch.nn.MaxPool2d(3, 2, padding=1)]
        p2 = [Residual(filters[1], filters[1], 1),
              Residual(filters[1], filters[1], 1)]
        p3 = [Residual(filters[1], filters[2], 2),
              Residual(filters[2], filters[2], 1)]
        p4 = [Residual(filters[2], filters[3], 2),
              Residual(filters[3], filters[3], 1)]
        p5 = [Residual(filters[3], filters[4], 2),
              Residual(filters[4], filters[4], 1)]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

        self.pool = torch.nn.AvgPool2d((8, 4), 1)
        self.fc = torch.nn.Sequential(torch.nn.Linear(512, 256),
                                      torch.nn.BatchNorm1d(256),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Dropout(),
                                      torch.nn.Linear(256, 751))

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x.div(x.norm(p=2, dim=1, keepdim=True))


class BasicBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = torch.nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(c_out)
        self.relu = torch.nn.ReLU(True)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, 3, stride=1,
                                     padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                                                  torch.nn.BatchNorm2d(c_out))
        elif c_in != c_out:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                                                  torch.nn.BatchNorm2d(c_out))
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return self.relu(x.add(y))


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return torch.nn.Sequential(*blocks)


class Net(torch.nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super().__init__()
        # 3 128 64
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                        torch.nn.BatchNorm2d(64),
                                        torch.nn.ReLU(inplace=True),
                                        torch.nn.MaxPool2d(3, 2, padding=1), )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = torch.nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = torch.nn.Sequential(torch.nn.Linear(512, 256),
                                              torch.nn.BatchNorm1d(256),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Dropout(),
                                              torch.nn.Linear(256, num_classes), )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.reid = torch.load(model_path,
                               map_location=torch.device(self.device))['model']
        self.reid.to(self.device)
        if torch.cuda.is_available():
            self.reid.half()
        self.size = (64, 128)
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(numpy.float32) / 255., size)

        batch = [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops]
        return torch.cat(batch, dim=0).float()

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            if torch.cuda.is_available():
                im_batch = im_batch.half()
            features = self.reid(im_batch)
        return features.cpu().numpy()


class DeepSort(object):
    def __init__(self,
                 model_path='weights/reid.pt',
                 max_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=0.5, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        metric = util.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = util.Tracker(metric,
                                    max_iou_distance=max_iou_distance,
                                    max_age=max_age, n_init=n_init)
        self.height, self.width = 1, 1

    def update(self, bbox_xywh, confidences, oids, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tl_wh = self._xywh_to_tlwh(bbox_xywh)
        detections = [util.Detection(bbox_tl_wh[i], conf, features[i], oid) for i, (conf, oid) in
                      enumerate(zip(confidences, oids)) if conf > self.min_confidence]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_oid = track.oid
            outputs.append(numpy.array([x1, y1, x2, y2, track_id, track_oid], dtype=numpy.int))
        if len(outputs) > 0:
            outputs = numpy.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(box_xy_wh):
        box_tl_wh = box_xy_wh.clone()
        box_tl_wh[:, 0] = box_xy_wh[:, 0] - box_xy_wh[:, 2] / 2.
        box_tl_wh[:, 1] = box_xy_wh[:, 1] - box_xy_wh[:, 3] / 2.
        return box_tl_wh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):

        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = numpy.array([])
        return features
