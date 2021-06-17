import json
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DetectionDataset(data.Dataset):
    def __init__(self, root, transforms=None, split='train',
                 train_size=0.9, include_text=False, include_filenames=False):
        super(DetectionDataset, self).__init__()
        self.root = Path(root)
        self.train_size = train_size
        self.image_names = []
        self.image_raw_boxes = []
        self.transforms = transforms
        self.include_text = include_text
        self.include_filenames = include_filenames
        self.image_texts = []

        if split in ['train', 'val']:
            plates_filename = self.root / 'train.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            # граница между train и valid
            train_valid_border = int(len(json_data) * train_size) + 1
            data_range = (0, train_valid_border) if split == 'train' \
                else (train_valid_border, len(json_data))
            # загружаем названия файлов и разметку
            self.load_data(json_data[data_range[0]:data_range[1]])
            return
        if split == 'test':
            plates_filename = self.root / 'submission.csv'
            self.load_test_data(plates_filename, split, train_size)
            return
        raise NotImplementedError(f'Unknown split: {split}')

    def load_data(self, json_data):
        for sample in json_data:
            if sample['file'] == 'train/25632.bmp':
                continue
            self.image_names.append(self.root / sample['file'])
            boxes = []
            texts = []
            for box in sample['nums']:
                boxes.append(np.array(box['box']))
                texts.append(box['text'])
            boxes = torch.as_tensor(boxes)
            self.image_raw_boxes.append(boxes)
            self.image_texts.append(texts)

    def load_test_data(self, plates_filename, split, train_size):
        df = pd.read_csv(plates_filename, usecols=['file_name'])
        for row in df.iterrows():
            self.image_names.append(self.root / row[1][0])
        self.image_boxes = None
        self.image_texts = None
        self.image_raw_boxes = None

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = {}
        if self.image_raw_boxes is not None:
            raw_boxes = self.image_raw_boxes[idx].clone()
            target['raw_boxes'] = raw_boxes
            target['boxes'] = self.get_rectangular_boxes(raw_boxes)
            target['labels'] = torch.ones((raw_boxes.shape[0])).long()
            target['masks'] = self.build_masks(raw_boxes, image)
            if self.include_text:
                target['text'] = self.image_texts[idx]
            if self.include_filenames:
                target['filename'] = self.image_names[idx]
        else:
            target['filename'] = self.image_names[idx]

        if self.transforms is not None:
            for transform in self.transforms:
                if transform[1] == 'image':
                    image = transform[0](image)
                elif transform[1] == 'sample':
                    image, target = transform[0](image, target)
                else:
                    raise NotImplementedError
        return image, target

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def get_rectangular_boxes(raw_boxes):
        rect_boxes = []
        for box in raw_boxes:
            x_0 = box[:, 0].min().reshape(-1)
            y_0 = box[:, 1].min().reshape(-1)
            x_1 = box[:, 0].max().reshape(-1)
            y_1 = box[:, 1].max().reshape(-1)
            rect_boxes.append(torch.cat((x_0, y_0, x_1, y_1), dim=0))
        return torch.stack(rect_boxes)

    @staticmethod
    def build_masks(raw_boxes, image):
        masks = []
        for box in raw_boxes:
            mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, box.numpy(), 1)
            masks.append(torch.tensor(mask))
        return torch.stack(masks)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def create_detection_model(rpn_anchor_generator=None):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=True,
        rpn_anchor_generator=rpn_anchor_generator,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):

        if random.random() < self.p:
            w = image.shape[1]
            image = cv2.flip(image, 1)

            flipped_masks = []
            for mask in target['masks']:
                mask = cv2.flip(mask.numpy(), 1)
                mask = torch.tensor(mask)
                flipped_masks.append(mask)
            target['masks'] = torch.stack(flipped_masks)

            flipped_boxes = []
            for box in target['boxes']:
                box[0], box[2] = w - box[2], w - box[0]
                flipped_boxes.append(box)
            target['boxes'] = torch.stack(flipped_boxes)

        return image, target


class PerspectiveTransform:
    def __init__(self, p=0.5, max_size_reduce=0.15):
        self.p = p
        self.max_size_reduce = max_size_reduce

    def __call__(self, image, target):
        # if random.random() < self.p:
        if random.random() > self.p:
            return image, target
        h, w, _ = image.shape
        # print(w, h)
        min_left = w
        max_right = 0
        min_top = h
        max_bottom = 0
        for box in target['boxes']:
            box = box.numpy()
            if box[0] < min_left:
                min_left = box[0]
            if box[1] < min_top:
                min_top = box[1]
            if box[2] > max_right:
                max_right = box[2]
            if box[3] > max_bottom:
                max_bottom = box[3]

        # Get points for perspective transform

        left_bar = np.floor(np.min([w * self.max_size_reduce,
                                    min_left]))
        right_bar = np.floor(np.min([w * self.max_size_reduce,
                                     w - max_right]))
        top_bar = np.floor(np.min([h * self.max_size_reduce,
                                   min_top]))
        bottom_bar = np.floor(np.min([h * self.max_size_reduce,
                                      h - max_bottom]))

        left = np.zeros(2)
        right = np.zeros(2)
        top = np.zeros(2)
        bottom = np.zeros(2)
        if left_bar > 0:
            left = np.random.randint(0, left_bar, 2)
        if top_bar > 0:
            top = np.random.randint(0, top_bar, 2)
        if right_bar > 0:
            right = np.random.randint(0, right_bar, 2)
        if bottom_bar > 0:
            bottom = np.random.randint(0, bottom_bar, 2)

        pts1 = np.float32([[left[0], top[0]],
                           [w - right[0], top[1]],
                           [left[1], h - bottom[0]],
                           [w - right[1], h - bottom[1]]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        # Make transformation

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, (w, h))

        warped_masks = []
        warped_boxes = []
        for mask in target['masks']:
            mask = cv2.warpPerspective(mask.numpy(), M, (w, h))
            mask = torch.tensor(mask)
            warped_masks.append(mask)

            # recalculate boxes
            cols = torch.sum(mask == 1, axis=0)
            rows = torch.sum(mask == 1, axis=1)

            i = 0
            while cols[i] == 0:
                i += 1
            x_0 = i
            i = cols.shape[0] - 1
            while cols[i] == 0:
                i -= 1
            x_1 = i
            i = 0
            while rows[i] == 0:
                i += 1
            y_0 = i
            i = rows.shape[0] - 1
            while rows[i] == 0:
                i -= 1
            y_1 = i
            warped_boxes.append(torch.tensor([x_0, y_0, x_1, y_1]))

        target['masks'] = torch.stack(warped_masks)
        target['boxes'] = torch.stack(warped_boxes)
        return dst, target
