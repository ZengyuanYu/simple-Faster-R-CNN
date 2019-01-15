#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
from pprint import pprint

class Config:
    # data
    voc_data_dir = '/media/yu/VOCdevkit/VOC2007/'
    min_size = 600  # resize
    max_size = 1000
    num_workers = 8
    test_num_workers = 8

    # l1_smooth_loss sigma
    rpn_sigma = 3.
    roi_sigma = 1.

    # 优化器参数
    # 论文中为0.0005
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    # 可视化
    env = 'faster-rcnn'
    port = 8097
    plot_every = 40

    #
    data = 'voc'
    pretrained_model = 'vgg16'

    # training epoch
    epoch = 14

    use_adam = False # 用Adama优化器
    use_chainer = False
    use_drop = False # use dropout 在ROIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    #model
    load_path = None

    caffe_pretrain = False
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise  ValueError('未知选项:"--%s"' % k)
            setattr(self, k, v)

        print('========user config=======')
        pprint(self._state_dict())
        print('---------end---------')

    def _state_dict(self):
        return {k: getattr(self, k) for k,_ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
