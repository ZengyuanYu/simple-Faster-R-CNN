#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
import numpy as np
from PIL import Image
import random


def read_image(path, dtype=np.float32, color=True):
    """
    从给定地址读取image
    :param path: 图片文件地址
    :param dtype: array类型
    :param color(bool): True:返回RGB
                        False:返回灰度图
    :return:image
    """
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        # 判断对象有'close'属性,bool返回
        if hasattr(f, 'close'):
            f.close

    if img.ndim == 2:
        # reshape (H,W) -> (1,H,W)
        return img[np.newaxis]
    else:
        # transpose (H,W,C) -> (C,H,W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """
    根据image size 来resize bbox
    :param bbox: (R,4) R:图片中Bbox的数量
                       4:y_min, x_min, y_max, x_max
    :param in_size: 元组,长为2,resize之前image的H和W
    :param out_size: 元组,长为2,resize之后image的H,W
    :return:rescale bbox
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def filp_bbox(bbox, size, y_filp=False, x_filp=False):
    """
    Filp bbox
    :param bbox: (R,4) R:图片中Bbox的数量
                       4:y_min, x_min, y_max, x_max
    :param size: 元组,长为2,resize之前image的H和W
    :param y_filp(bool): 根据垂直翻转图像来翻转bbox
    :param x_filp(bool): 根据水平翻转图像来翻转bbox
    :return: 翻转后的bbox
    """
    H, W = size
    bbox = bbox.copy()
    if y_filp:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_filp:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def crop_bbox(bbox, y_slice=None, x_slice=None,
              allow_outside_center=True,return_param=False
              ):
    """

    :param bbox:(R,4) R:图片中Bbox的数量
                       4:y_min, x_min, y_max, x_max
    :param y_slice: y轴切片
    :param x_slice: x轴切片
    :param allow_outside_center(bool):True:bbox的中心点在裁剪外,则移除bbox
    :param return_param(bool):True:返回bbox的索引
    :return: (bbox, dict) or (bbox)
    """
    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))
    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        # np.logical_and([True, False], [False, False])
        # -> array([False, False])
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, :2] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        # 返回非0index
        # >>> np.flatnonzero(array([-2, -1,  0,  1,  2]))
        # array([0, 1, 3, 4])
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf    # np.inf为一个无限大数
    if slice_.start is None:
        l = 0
    else:
        l = slice_.start
    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """
    转换bbox
    :param bbox:(R,4) R:图片中Bbox的数量
                       4:y_min, x_min, y_max, x_max
    :param y_offset: int or float, y轴的抵消部分
    :param x_offset: int or float, x轴抵消部分
    :return: 通过给定抵消转换后的bbox
    """
    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_filp(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """
    在垂直和水平方向随机翻转image
    :param img: (C,H,W)格式
    :param y_random(bool):True:垂直翻转
    :param x_random(bool):True:水平翻转
    :param return_param(bool):True:返回翻转信息
    :param copy(bool):True:但会image副本
    :return:
    """
    y_filp, x_filp = False, False
    if y_random:
        y_filp = random.choice([True, False])
    if x_random:
        x_filp = random.choice([True, False])

    # a='python'
    # b=a[::-1]
    # print(b) #nohtyp
    if y_filp:
        img = img[:, ::-1, :]
    if x_filp:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_filp, 'x_flip': x_filp}
    else:
        return img


