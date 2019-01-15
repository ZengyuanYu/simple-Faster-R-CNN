#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image


class VOCBboxDataset:

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """
        返回第i个example,返回一个彩色图片和bboxes, 图片是CHW类型,RGB
        :param i: int
        :return: 元组, image 和 bboxes
        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        # list(tuple) -> 将元组转为列表
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # 当不用'difficult'分割, 则此目标是'difficult',跳过
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # 当use_difficult=False,则所有在'difficult'元素都是False
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        # 载入图片
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
