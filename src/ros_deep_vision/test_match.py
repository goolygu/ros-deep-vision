#! /usr/bin/env python
# -*- coding: utf-8

import sys
import os
import cv2
import numpy as np
import time
import StringIO
from threading import Lock

from misc import WithTimer
from numpy_cache import FIFOLimitedArrayCache
from app_base import BaseApp
from core import CodependentThread
from image_misc import *

from time import gmtime, strftime
from utils import DescriptorHandler, Descriptor
import settings
import caffe
import re

class Tester:

    def __init__(self, settings):
        print 'initialize'
        self.settings = settings
        self.descriptor_layers = ['conv5','conv4']

        self._data_mean = np.load(settings.caffevis_data_mean)
        # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
        excess_h = self._data_mean.shape[1] - self.settings.caffevis_data_hw[0]
        excess_w = self._data_mean.shape[2] - self.settings.caffevis_data_hw[1]
        assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(self.settings.caffevis_data_hw)
        self._data_mean = self._data_mean[:, excess_h:(excess_h+self.settings.caffevis_data_hw[0]),
                                          excess_w:(excess_w+self.settings.caffevis_data_hw[1])]
        self._net_channel_swap = (2,1,0)
        self._net_channel_swap_inv = tuple([self._net_channel_swap.index(ii) for ii in range(len(self._net_channel_swap))])
        self._range_scale = 1.0      # not needed; image comes in [0,255]



        if settings.caffevis_mode_gpu:
            caffe.set_mode_gpu()
            print 'CaffeVisApp mode: GPU'
        else:
            caffe.set_mode_cpu()
            print 'CaffeVisApp mode: CPU'

        self.net = caffe.Classifier(
            settings.caffevis_deploy_prototxt,
            settings.caffevis_network_weights,
            mean = self._data_mean,
            channel_swap = self._net_channel_swap,
            raw_scale = self._range_scale,
            #image_dims = (227,227),
        )
        self.input_dims = self.net.blobs['data'].data.shape[2:4]    # e.g. (227,227)
        self.img_file_list = []

        self.descriptor_handler = DescriptorHandler(self.settings.ros_dir + '/eric_data_set/model_seg/', self.descriptor_layers)

    def run(self,dir):
        self.init_img_file_list(dir)
        for img_file in self.img_file_list:
            # print "test:  ", img_file
            match_file = self.classify(dir+img_file)
            print img_file +":"+ match_file
            # self.classify(dir+self.img_file_list[0])

        pass

    def net_preproc_forward(self, net, img):
        assert img.shape == (227,227,3), 'img is wrong size'
        #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)
        data_blob = net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        output = net.forward(data=data_blob)
        return output

    def classify(self, img_file):

        image = cv2_read_file_rgb(img_file)
        image = crop_to_square(image)
        image = cv2.resize(image, self.input_dims)
        self.net_preproc_forward(self.net, image)
        desc_current = self.descriptor_handler.gen_descriptor('current', self.net.blobs)
        match_file = self.descriptor_handler.get_max_match(desc_current)
        return match_file

    def init_img_file_list(self,dir):
        match_flags = re.IGNORECASE
        for filename in os.listdir(dir):
            if re.match('.*\.(jpg|jpeg|png)$', filename, match_flags):
                self.img_file_list.append(filename)
        self.img_file_list = sorted(self.img_file_list)

    def create_desc_files(self,dir):
        self.init_img_file_list(dir)

        for file in self.img_file_list:
            print 'file ' + file
            image = cv2_read_file_rgb(dir+file)
            image = crop_to_square(image)
            image = cv2.resize(image, self.input_dims)
            self.net_preproc_forward(self.net, image)
            tag_name = file.split('.')[0]

            desc = self.descriptor_handler.gen_descriptor(tag_name, self.net.blobs)
            self.descriptor_handler.save(dir, desc)


if __name__ == '__main__':
    test = Tester(settings)
    # test.create_desc_files(settings.ros_dir + '/eric_data_set/model_seg/')
    test.run(settings.ros_dir + '/eric_data_set/test_seg/')
