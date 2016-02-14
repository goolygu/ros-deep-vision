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
# from core import CodependentThread
from image_misc import *

from time import gmtime, strftime
from utils import DescriptorHandler, Descriptor
import settings
import caffe
import re
from collections import namedtuple
import yaml
from data_collector import Data
from data_ploter import *
import pcl

class DataMonster:

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
        self.available_layer = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']



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

        # self.resize_ratio = 280/self.input_dims[0]
        self.xy_bias = get_crop_bias()
        # self.data_file_list = []

        # self.descriptor_handler = DescriptorHandler(self.settings.ros_dir + '/eric_data_set/model_seg/', self.descriptor_layers)

    def run(self,dir):

        data_list = self.get_data_all(dir)

        # data_list = data_list[0:5]

        conv5_list = self.load_conv5(data_list, dir)

        print "conv5", conv5_list.shape

        # 280/13
        self.resize_ratio = get_after_crop_size()[0] / conv5_list.shape[2]
        self.receptive_field_size = int(round(self.resize_ratio))
        self.receptive_grid = self.gen_receptive_grid(self.receptive_field_size)

        hist = np.zeros(conv5_list.shape[1])
        max_hist = np.zeros(conv5_list.shape[1])
        for idx, conv5 in enumerate(conv5_list):
            # print idx
            bin_data, max_data = self.binarize(conv5)
            hist = hist + bin_data
            max_hist = max_hist + max_data
        print "hist", hist
        filter_idx_list = np.argsort(hist)[::-1]
        print filter_idx_list[0:20]
        filter_idx_list = filter_idx_list[0:5]
        # print filter_idx_list
        # dist_list = [ [] for f in filter_idx_list]
        dist_list = np.empty([len(filter_idx_list),len(data_list),3])

        for idx, conv5 in enumerate(conv5_list):
            print idx
            data = data_list[idx]
            name = data.name
            max_idx_list = []
            max_xy_list = []
            for filter_idx in filter_idx_list:
                max_idx = np.argmax(conv5[filter_idx], axis=None)
                # print "max idx", max_idx
                max_xy = np.unravel_index(max_idx, conv5.shape[1:3])
                # print "max", max_xy
                orig_xy = self.get_orig_xy(max_xy)
                # print "orig", orig_xy

                max_xy_list.append(orig_xy)

                # orig_idx = np.ravel_multi_index(orig_xy,(480,640))
                # max_idx_list.append(orig_idx)
            max_xyz_list = self.get_average_xyz_from_point_cloud(dir, name, max_xy_list)
            # max_xyz_list = self.get_xyz_from_point_cloud(dir, name, max_idx_list)
            palm_xyz = np.array(self.get_palm_xyz(data))
            diff_list = [palm_xyz - feature_xyz for feature_xyz in max_xyz_list]
            dist_list[:,idx,:] = np.array(diff_list)
            # dist_list.append(diff_list)
            # for filter_idx, diff in enumerate(diff_list):
            #     dist_list[filter_idx].append(diff)

        for i, filter_idx in enumerate(filter_idx_list):
            plot_dist(dist_list[i,:,:])


    def net_preproc_forward(self, img):
        assert img.shape == (227,227,3), 'img is wrong size'
        #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)
        data_blob = self.net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        output = self.net.forward(data=data_blob)
        return output

    def net_proc_forward_layer(self, img, mask):
        assert img.shape == (227,227,3), 'img is wrong size'

        data_blob = self.net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        # print "mask", mask.shape

        for idx in range(len(self.available_layer)-1):
            output = self.net.forward(data=data_blob,start=self.available_layer[idx],end=self.available_layer[idx+1])
            if self.available_layer[idx].startswith("conv"):
                new_blob = self.net.blobs[self.available_layer[idx]].data
                new_blob.data = self.mask_out(self.net.blobs[self.available_layer[idx]].data, mask)
                self.net.blobs[self.available_layer[idx]] = new_blob
            # print output
        return output

    def get_orig_xy(self, xy):
        new_x = int(round(xy[0]*self.resize_ratio + self.xy_bias[0]))
        new_y = int(round(xy[1]*self.resize_ratio + self.xy_bias[1]))
        return (new_x, new_y)

    def get_palm_xyz(self, data):
        return data.pose_dict["r2/left_palm"][0]

    def gen_receptive_grid(self, receptive_field_size):
        return np.mgrid[0:receptive_field_size,0:receptive_field_size]
        # x = np.arange(0,receptive_field_size)
        # y = np.arange(0,receptive_field_size)
        # xv, yv = np.meshgrid(x,y)
        # return (xv, yv)

    def get_average_xyz_from_point_cloud(self, dir, name, max_xy_list):
        p = pcl.PointCloud()
        p.from_file(dir + name + ".pcd")
        a = np.asarray(p)
        output = []
        for xy in max_xy_list:
            grid = np.zeros(self.receptive_grid.shape)
            grid[0] = xy[0] + self.receptive_grid[0]
            grid[1] = xy[1] + self.receptive_grid[1]
            xy_receptive_list =np.reshape(grid, [2,-1])
            idx_receptive_list = np.ravel_multi_index(xy_receptive_list.astype(int),(480,640))
            avg = np.nanmean(a[idx_receptive_list],axis=0)
            # avg = np.nanmedian(a[idx_receptive_list],axis=0)
            output.append(avg)

        return np.array(output)

    def get_xyz_from_point_cloud(self, dir, name, max_idx_list):
        p = pcl.PointCloud()
        p.from_file(dir + name + ".pcd")
        a = np.asarray(p)
        return a[max_idx_list]
        # print "pc shape", a.shape
        # for idx in max_idx_list:#range(1,len(a),30):#
        #     print a[idx]

    def mask_out(self, data, mask):
        # print "data shape", data.shape
        dim = data.shape

        for y in range(dim[2]):
            for x in range(dim[3]):
                if is_masked((dim[2],dim[3]),(x,y),mask):
                    data[:,:,y,x] = 0

        return data
    #
    # def classify(self, img_file):
    #
    #     image = cv2_read_file_rgb(img_file)
    #     image = crop_to_square(image)
    #     image = cv2.resize(image, self.input_dims)
    #     self.net_preproc_forward(self.net, image)
    #     desc_current = self.descriptor_handler.gen_descriptor('current', self.net.blobs)
    #     match_file = self.descriptor_handler.get_max_match(desc_current)
    #     return match_file

    def get_data_all(self,dir):
        data_file_list = []
        match_flags = re.IGNORECASE
        for filename in os.listdir(dir):
            if re.match('.*_data\.yaml$', filename, match_flags):
                data_file_list.append(filename)
        data_file_list = sorted(data_file_list)

        # data_dict = {}
        data_list = []

        for data_file in data_file_list:
            f = open(dir+data_file)
            data = yaml.load(f)
            data_list.append(data)
            # data_dict[data.name] = data
        return data_list

    def load_conv5(self, data_list, dir):

        conv5_list = np.array([]).reshape([0] + list(self.net.blobs['conv5'].data.shape[1:]))

        for idx, data in enumerate(data_list):
            print idx

            img_name = dir + data.name + "_rgb.png"
            img = cv2_read_file_rgb(img_name)
            img = crop_to_center(img)

            mask_name = dir + data.name + "_mask.png"
            mask = cv2.imread(mask_name)
            mask = crop_to_center(mask)
            mask = np.reshape(mask[:,:,0], (mask.shape[0], mask.shape[1]))
            # print "shape img", img.shape
            # print "shape mask", mask.shape

            img = cv2.resize(img, self.input_dims)

            # print "img", img.shape
            self.net_proc_forward_layer(img, mask)
            # self.net_preproc_forward(img)
            conv5_list = np.append(conv5_list, self.net.blobs['conv5'].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv5_list

    def binarize(self, data):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        max_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            max_value = np.amax(filter)
            if max_value > 20:
                bin_data[id] = 1
            max_data[id] = max_value
        return bin_data, max_value

    def average(self, data):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            sum_value = np.sum(filter)
            bin_data[id] = sum_value / (data.shape[1]*data.shape[2])

        return bin_data

if __name__ == '__main__':

    data_monster = DataMonster(settings)
    data_monster.run(settings.ros_dir + '/data/')
