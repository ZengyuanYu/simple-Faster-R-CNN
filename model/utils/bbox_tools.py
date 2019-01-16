#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
import numpy as np
import numpy as xp

import six
from six import __init__


def loc2bbox(src_bbox, loc):
    """
    从边界框偏移(offset)和比例(scale)解码边界框
    :param src_bbox: (R,4) p_ymin, p_xmin, p_ymax, p_xmax
    :param loc: t_y, t_x, t_h, t_w
    :return: (R,4)
    """
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    根据源bbox和目标bbox,求出loc
    :param src_bbox: (R,4) p_ymin, p_xmin, p_ymax, p_xmax
    :param dst_bbox: (R,4) g_ymin, g_xmin, g_ymax, g_xmax
    :return: loc (R,4) t_y, t_x, t_h, t_w
    """
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 1] + 0.5 * width
    ctr_y = src_bbox[:, 0] + 0.5 * height

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = src_bbox[:, 1] + 0.5 * base_width
    base_ctr_y = src_bbox[:, 0] + 0.5 * base_height

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    计算两个bbox的IOU
    :param bbox_a: (N,4)
    :param bbox_b: (K,4)
    :return: array (N,K) 包含N个bbox_a与K个bbox_b的IOU
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # 左上
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # 右下
    br = xp.maximum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    # a = numpy.array([1,2,3,4])
    # >>> numpy.prod(a) = 24 1*2*3*4
    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def __test():
    pass


if __name__ == '__main__':
    __test()


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """
    遍历ratios,scales,生成基本的anchor
    :param base_size: 基础参考window的宽和高
    :param ratios:
    :param anchor_scales:
    :return: (R,4)
            R = len(ratios) * len(anchor_scales)
            y_min, x_min, y_max, x_max
    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
