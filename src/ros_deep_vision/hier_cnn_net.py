#! /usr/bin/env python
# -*- coding: utf-8
import roslib
import rospy
import sys
import os
import cv2
import numpy as np
import time
import StringIO
from threading import Lock

from misc import WithTimer

from image_misc import *

from time import gmtime, strftime
import settings
import caffe

from distribution import *

import copy
import scipy
from scipy.special import expit

from data_util import *

from backprop_info import *

class HierCNNNet:

    def __init__(self, settings, data_settings):
        print 'initialize'
        self.settings = settings

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

        GPU_ID = 0
        if settings.caffevis_mode_gpu:
            caffe.set_mode_gpu()
            print 'CaffeVisApp mode: GPU'
            caffe.set_device(GPU_ID)
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

        self.set_data_settings(data_settings)
        self.visualize = True
        self.show_backprop = True

        self.backprop_info = BackPropInfo(data_settings)

    def set_data_settings(self, data_settings):

        self.ds = data_settings

        self.back_mode = self.ds.back_prop_mode
        w = self.ds.avg_pointcloud_width
        self.average_grid = np.mgrid[-w:w,-w:w]

    # layer_diff_list: list of diff for each data for back propagation
    # backprop info
    def find_consistent_filters_recursive(self, data_list, layer_diff_list, distribution, backprop_info, parent_filters):

        layer_name = backprop_info.get_layer_name()
        next_layer_name = backprop_info.get_next_layer_name()
        num_filters = backprop_info.get_num_filters(layer_name)
        frame_list = backprop_info.get_frame_list(layer_name)

        if next_layer_name is None:
            return

        filter_idx_list = self.find_consistent_filters(layer_diff_list, 0., num_filters)
        print "consistent filter", layer_name, filter_idx_list

        for filter_idx in filter_idx_list:

            print "handling filter", filter_idx, "layer", layer_name
            # img_src is the abs diff back propagate to the image layer
            next_layer_diff_list, img_src_list = self.load_layer_fix_filter_list(next_layer_name, layer_name, layer_diff_list, data_list, filter_idx)

            rel_pos = self.get_relative_pos(filter_idx, data_list, layer_diff_list, img_src_list, frame_list, 0.)
            self.set_distribution(distribution, frame_list, filter_idx, rel_pos, parent_filters)
            next_backprop_info = copy.deepcopy(backprop_info)
            next_backprop_info.pop()
            next_parent_filters = copy.deepcopy(parent_filters)
            next_parent_filters.append(filter_idx)
            self.find_consistent_filters_recursive(data_list, next_layer_diff_list, distribution, next_backprop_info, next_parent_filters)

    def train(self, data_list):
        distribution = Distribution()

        conv_list = self.load_layer(data_list, self.backprop_info.get_layer_name())

        self.find_consistent_filters_recursive(data_list, conv_list, distribution, self.backprop_info, [])

        return distribution


    def set_distribution(self, distribution, frame_list, filter_idx, rel_pos_list, parent_filters):
        for j, frame in enumerate(frame_list):
            distribution.set(parent_filters + [filter_idx], frame, rel_pos_list[j,:,:])
        distribution.set_tree(parent_filters, filter_idx)


    def get_filter_xyz(self, layer_data, pc, threshold):
        resize_ratio = float(pc.shape[0]) / float(layer_data.shape[0])
        if True:
            filter_xy = self.get_filter_avg_xy(layer_data, threshold)
        elif False:
            filter_xy = self.get_filter_avg_xy_square(layer_data, threshold)
        elif False:
            filter_xy = self.get_filter_avg_xy_thresholded(layer_data, threshold)
        else:
            filter_xy = self.get_filter_max_xy(layer_data, threshold)
        orig_xy = self.get_orig_xy(filter_xy, resize_ratio)
        if self.ds.xy_to_cloud_xyz == "avg":
            filter_xyz = self.get_average_xyz_from_point_cloud(pc, orig_xy, self.average_grid)
        elif self.ds.xy_to_cloud_xyz == "closest":
            filter_xyz = self.get_closest_xyz_from_point_cloud(pc, orig_xy, max_width = self.ds.avg_pointcloud_width)
        return filter_xyz, filter_xy


    def get_all_filter_xyz_recursive(self, data, conv_data, parent_filters, backprop_info, filter_tree=None, idx=0):
        xyz_dict = {}
        xy_dict = {}
        response_dict = {}

        layer_name = backprop_info.get_layer_name()
        next_layer_name = backprop_info.get_next_layer_name()
        num_filters = backprop_info.get_num_filters(layer_name)
        if next_layer_name is None:
            return xyz_dict, response_dict

        filter_idx_list = []
        if filter_tree is None:
            filter_idx_list = self.get_top_filters(conv_data, num_filters, None)
        else:
            filter_idx_list = filter_tree.keys()

        # if self.ds.filter_test == 'top':
        #     if filter_tree is None:
        #         filter_idx_list = self.get_top_filters(conv_data, num_filters, None)
        #     else:
        #         filter_idx_list = self.get_top_filters(conv_data, num_filters, filter_tree.keys())
        # elif self.ds.filter_test == 'all':
        #     filter_idx_list = filter_tree.keys()

        for filter_idx in filter_idx_list:
            print parent_filters + (filter_idx,)

            if not self.filter_response_pass_threshold(conv_data[filter_idx], 0):
                continue

            conv_diff, img_src, img_src_color = self.load_layer_fix_filter(next_layer_name, layer_name, conv_data, data, filter_idx)

            filter_id = parent_filters + (filter_idx,)

            xyz_dict[filter_id], max_xy = self.get_filter_xyz(img_src, data.pc, 0)
            response_dict[filter_id] = self.get_max_filter_response(conv_data[filter_idx])
            xy_dict[filter_id] = max_xy

            if False and filter_id in [(23,), (23,60,)]:
                self.show_grad_compare(str(idx) + "_" + str(filter_id) + "_" + data.name, img_src_color, data, max_xy)

            self.show_gradient(str(filter_id), self.net.blobs['data'], max_xy, 0)
            next_backprop_info = copy.deepcopy(backprop_info)
            next_backprop_info.pop()
            if filter_tree is None:
                next_filter_tree = None
            else:
                next_filter_tree = filter_tree[filter_idx]
            sub_xyz_dict, sub_xy_dict, sub_response_dict = self.get_all_filter_xyz_recursive(data, conv_diff, filter_id, next_backprop_info, next_filter_tree, idx)

            xyz_dict.update(sub_xyz_dict)
            xy_dict.update(sub_xy_dict)
            response_dict.update(sub_response_dict)

        return xyz_dict, xy_dict, response_dict

    # def get_all_filter_xyz(self, data, dist):
    #
    #     self.net_proc_forward_layer(data.img, data.mask)
    #     layer = self.backprop_info.get_layer_name()
    #     conv5_data = copy.deepcopy(self.net.blobs[layer].data[0,:])
    #
    #     xyz_dict, xy_dict, response_dict = self.get_all_filter_xyz_recursive(data, conv5_data, (), self.backprop_info, dist.filter_tree)
    #
    #     return xyz_dict, response_dict

    # dist is expected features
    def get_all_filter_xyz(self, data, dist, idx=0):

        self.net_proc_forward_layer(data.img, data.mask)
        layer = self.backprop_info.get_layer_name()
        conv5_data = copy.deepcopy(self.net.blobs[layer].data[0,:])

        if dist is None:
            xyz_dict, xy_dict, response_dict = self.get_all_filter_xyz_recursive(data, conv5_data, (), self.backprop_info, None, idx)
        else:
            xyz_dict, xy_dict, response_dict = self.get_all_filter_xyz_recursive(data, conv5_data, (), self.backprop_info, dist.filter_tree, idx)

        return xyz_dict, xy_dict, response_dict


    def show_grad_compare(self, name, img_src_color, data, max_xy):
        grad_img = norm0255(img_src_color[:,:,(2,1,0)])
        img = data.img[:,:,(2,1,0)]

        for i in range(-3,3):
            for j in range(-3,3):
                grad_img[max_xy[0]+i][max_xy[1]+j] = [0,0,255]

        cv2.imshow(name, np.concatenate((grad_img, img), axis = 1))
        cv2.waitKey(200)
        # save image
        # cv2.imwrite(os.path.join(os.path.expanduser('~'), name + ".png"), np.concatenate((grad_img, img), axis = 1))

    def show_gradient(self, name, data_layer, xy_dot=(0,0), threshold=0):
        if not self.visualize or not self.show_backprop:
            return
        grad_blob = data_layer.diff
        grad_blob = grad_blob[0]                    # bc01 -> c01
        grad_blob = grad_blob.transpose((1,2,0))    # c01 -> 01c
        grad_img = grad_blob[:, :, (2,1,0)]  # e.g. BGR -> RGB

        img_blob = data_layer.data
        img = img_blob[0].transpose((1,2,0))    # c01 -> 01c
        # grad_img2 = cv2.GaussianBlur(grad_img2, (5,5), 0)
        # grad_img = cv2.bilateralFilter(grad_img,9,75,75)

        # grad_img2 = np.absolute(grad_img).mean(axis=2)
        # xy_dot2 = self.get_filter_avg_xy(grad_img2, threshold) #
        # print xy_dot2

        # max_idx = np.argmax(grad_img.mean(axis=2))
        # xy_dot2 = np.unravel_index(max_idx, grad_img.mean(axis=2).shape)
        # xy_dot2 = self.get_filter_avg_xy(np.absolute(grad_img).mean(axis=2), threshold) #
        # print xy_dot2
        # xy_dot2 = xy_dot2.astype(int)
        # Mode-specific processing

        back_filt_mode = 'raw'#'norm'#
        if back_filt_mode == 'raw':
            grad_img = norm01c(grad_img, 0)
            # grad_img = fix01(grad_img)
        elif back_filt_mode == 'gray':
            grad_img = grad_img.mean(axis=2)
            grad_img = norm01c(grad_img, 0)
        elif back_filt_mode == 'norm':
            grad_img = np.linalg.norm(grad_img, axis=2)
            grad_img = norm01(grad_img)
        else:
            grad_img = np.linalg.norm(grad_img, axis=2)
            cv2.GaussianBlur(grad_img, (0,0), self.settings.caffevis_grad_norm_blur_radius, grad_img)
            grad_img = norm01(grad_img)

        # If necessary, re-promote from grayscale to color
        if len(grad_img.shape) == 2:
            grad_img = np.tile(grad_img[:,:,np.newaxis], 3)

        if not np.isnan(xy_dot[0]) and not np.isnan(xy_dot[1]):
            for i in range(-3,3):
                for j in range(-3,3):
                    grad_img[i+xy_dot[0],j+xy_dot[1]] = [1,0,0]

        # if not np.isnan(xy_dot2[0]) and not np.isnan(xy_dot2[1]):
        #     for i in range(-3,3):
        #         for j in range(-3,3):
        #             grad_img[i+xy_dot2[0],j+xy_dot2[1]] = [1,0,1]

        cv2.imshow(name, grad_img)
        # cv2.imwrite(self.path + "visualize/" + name + "_grad.png", norm0255(grad_img))
        cv2.waitKey(100)

    def find_consistent_filters(self, conv_list, threshold, number):
        if number <= 0:
            return []

        hist = np.zeros(conv_list.shape[1])
        max_hist = np.zeros(conv_list.shape[1])
        max_sum = np.zeros(conv_list.shape[1])
        for idx, conv in enumerate(conv_list):
            # print idx

            bin_data, max_data = self.binarize(conv, threshold)
            hist = hist + bin_data
            max_hist = np.amax(np.concatenate((max_hist[np.newaxis,...],max_data[np.newaxis,...]),axis=0),axis=0)

            if self.ds.top_filter == 'max':
                max_data = scipy.special.expit(max_data)
            elif self.ds.top_filter == 'maxlog':
                max_data = np.log(10*max_data+1)
            max_sum = max_data + max_sum

        # print "hist", hist
        # print "max hist", max_hist
        if self.ds.top_filter == 'above':
            filter_idx_list = np.argsort(hist)[::-1]
            for i in range(number+1):
                if hist[filter_idx_list[i]] == 0:
                    number = i
                    break
        elif self.ds.top_filter == 'max' or self.ds.top_filter == 'maxlog':
            filter_idx_list = np.argsort(max_sum)[::-1]
            for i in range(number+1):
                if max_sum[filter_idx_list[i]] <= 0:
                    number = i
                    break
        # print "top filters counts", hist[filter_idx_list[0:number+10]]
        print "top filters", filter_idx_list[0:number+10]
        print "max sum", max_sum[filter_idx_list[0:number+10]]

        return filter_idx_list[0:number]


    def get_filter_avg_xy(self, filter_response, threshold):
        # print "max", np.amax(filter_response)
        assert np.all(filter_response >= 0)
        # if self.back_mode == 'grad':
        #     filter_response = np.maximum(filter_response,0) #TODO: check if necessary
        # elif self.back_mode == 'deconv':
        #     filter_response = np.abs(filter_response)
        assert filter_response.ndim == 2, "filter size incorrect"

        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])

        xy_grid = np.mgrid[0:filter_response.shape[0], 0:filter_response.shape[0]]

        if np.sum(filter_response) == 0:
            return np.array([float('nan'),float('nan')])

        filter_response_norm = filter_response / float(np.sum(filter_response))
        avg_x = np.sum(xy_grid[0] * filter_response_norm)
        avg_y = np.sum(xy_grid[1] * filter_response_norm)

        return np.around(np.array([avg_x, avg_y])).astype(int)

    def get_filter_avg_xy_thresholded(self, filter_response, threshold):
        # print "max", np.amax(filter_response)
        filter_response = np.maximum(filter_response,0) #TODO: check if necessary
        assert filter_response.ndim == 2, "filter size incorrect"

        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])

        xy_grid = np.mgrid[0:filter_response.shape[0], 0:filter_response.shape[0]]

        if np.sum(filter_response) == 0:
            return np.array([float('nan'),float('nan')])

        filter_response[filter_response < max_value * 0.8] = 0

        filter_response_norm = filter_response / float(np.sum(filter_response))
        avg_x = np.sum(xy_grid[0] * filter_response_norm)
        avg_y = np.sum(xy_grid[1] * filter_response_norm)

        return np.around(np.array([avg_x, avg_y])).astype(int)

    def get_filter_avg_xy_square(self, filter_response, threshold):
        # print "max", np.amax(filter_response)
        filter_response = np.maximum(filter_response,0)
        assert filter_response.ndim == 2, "filter size incorrect"

        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])

        xy_grid = np.mgrid[0:filter_response.shape[0], 0:filter_response.shape[0]]

        if np.sum(filter_response) == 0:
            return np.array([float('nan'),float('nan')])
        filter_response = filter_response**2
        filter_response_norm = filter_response / float(np.sum(filter_response))
        avg_x = np.sum(xy_grid[0] * filter_response_norm)
        avg_y = np.sum(xy_grid[1] * filter_response_norm)

        return np.around(np.array([avg_x, avg_y])).astype(int)

    def get_filter_max_xy(self, filter_response, threshold):
        assert filter_response.ndim == 2, "filter size incorrect"
        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])
        max_idx = np.argmax(filter_response, axis=None)
        max_xy = np.unravel_index(max_idx, filter_response.shape)
        return max_xy

    def get_max_filter_response(self, filter_response):
        return np.nanmax(filter_response).item()

    def filter_response_pass_threshold(self, filter_response, threshold):
        max_v = np.nanmax(filter_response)
        # print "max", max_v
        if max_v <= threshold:
            print "failed threshold", max_v
            return False
        else:
            return True

    # filter_list: restrict to filters in list
    def get_top_filters(self, layer_response, max_num_filter, filter_list=None):
        if filter_list is None:
            num_filter = layer_response.shape[0]
            filter_list = np.arange(num_filter)
        else:
            num_filter = len(filter_list)
            filter_list = np.array(filter_list)
        max_list = np.zeros(num_filter)

        for i, filter_id in enumerate(filter_list):
            max_list[i] = np.amax(layer_response[filter_id])

        sorted_idx_list = np.argsort(max_list)[::-1]
        sorted_idx_list = sorted_idx_list[0:max_num_filter]

        sorted_filter_idx_list = filter_list[sorted_idx_list]

        return sorted_filter_idx_list.tolist()


    def get_relative_pos(self, filter_idx, data_list, conv_list, img_src, frame_list, threshold):

        relative_pos = np.empty([len(frame_list),len(data_list),3])

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()

            if not self.filter_response_pass_threshold(conv_list[idx][filter_idx], threshold):
                feature_xyz = (float('nan'),float('nan'),float('nan'))
                # continue
            else:
                if self.ds.location_layer == "image":
                    feature_xyz, filter_xy = self.get_filter_xyz(img_src[idx], data.pc, threshold)
                else:
                    feature_xyz, filter_xy = self.get_filter_xyz(abs(conv_list[idx][filter_idx]), data.pc, threshold)


            for frame_idx, frame in enumerate(frame_list):
                frame_xyz = np.array(self.get_frame_xyz(data, frame))
                diff = frame_xyz - feature_xyz
                relative_pos[frame_idx,idx,:] = diff

        return relative_pos

    # returns a distribution list of shape(num_frames, number of filters, num of data, 3)
    # distribution contains diff of frame xyz to feature xyz
    # conv_list is for checking if pass threshold
    def get_relative_pos_list(self, filter_idx_list, data_list, conv_list, img_src_fid_array, frame_list, threshold):

        relative_pos_list = np.empty([len(frame_list),len(filter_idx_list),len(data_list),3])

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()
            xyz_list = []

            for i, filter_idx in enumerate(filter_idx_list):
                # cv2.imshow(layer + " " + str(idx)+" "+ str(filter_idx),norm01c(bp[i], 0))
                # cv2.waitKey(200)
                if not self.filter_response_pass_threshold(conv_list[idx][filter_idx], threshold):
                    xyz_list.append((float('nan'),float('nan'),float('nan')))
                    continue

                if self.ds.location_layer == "image":
                    xyz, filter_xy = self.get_filter_xyz(img_src_fid_array[idx][i], data.pc, threshold)
                else:
                    xyz, filter_xy = self.get_filter_xyz(abs(conv_list[idx][filter_idx]), data.pc, threshold)
                xyz_list.append(xyz)

            for frame_idx, frame in enumerate(frame_list):
                frame_xyz = np.array(self.get_frame_xyz(data, frame))
                diff_list = [frame_xyz - feature_xyz for feature_xyz in xyz_list]
                relative_pos_list[frame_idx,:,idx,:] = np.array(diff_list)

        return relative_pos_list


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

        mode = 2
        # only mask out conv1
        if mode == 0:
            self.net.blobs['data'].data[...] = data_blob
            self.net.forward_from_to(start='conv1',end='relu1')
            self.mask_out(self.net.blobs['conv1'].data, mask)
            self.net.forward_from_to(start='relu1',end='prob')
        # mask out all conv layers
        elif mode == 1:
            self.net.blobs['data'].data[...] = data_blob
            self.net.forward_from_to(start='conv1',end='relu1')
            self.mask_out(self.net.blobs['conv1'].data, mask)
            self.net.forward_from_to(start='relu1',end='conv2')

            self.net.forward_from_to(start='conv2',end='relu2')
            self.mask_out(self.net.blobs['conv2'].data, mask)
            self.net.forward_from_to(start='relu2',end='conv3')

            self.net.forward_from_to(start='conv3',end='relu3')
            self.mask_out(self.net.blobs['conv3'].data, mask)
            self.net.forward_from_to(start='relu3',end='conv4')

            self.net.forward_from_to(start='conv4',end='relu4')
            self.mask_out(self.net.blobs['conv4'].data, mask)
            self.net.forward_from_to(start='relu4',end='conv5')

            self.net.forward_from_to(start='conv5',end='relu5')
            self.mask_out(self.net.blobs['conv5'].data, mask)
            self.net.forward_from_to(start='relu5',end='pool5')
        # mask out all conv layers, identical to mode 1
        elif mode == 2:
            for idx in range(len(self.available_layer)-1):
                output = self.net.forward(data=data_blob,start=self.available_layer[idx],end=self.available_layer[idx+1])
                if self.ds.mask == "mask" and self.available_layer[idx].startswith("conv"):
                    self.mask_out(self.net.blobs[self.available_layer[idx]].data, mask)


    def net_proc_backward(self, filter_idx, backprop_layer):

        diffs = self.net.blobs[backprop_layer].diff * 0
        diffs[0][filter_idx] = self.net.blobs[backprop_layer].data[0,filter_idx]

        assert self.back_mode in ('grad', 'deconv')
        if self.back_mode == 'grad':
            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
        else:
            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)

    # Set the backprop layer of filter_idx to data and backprop
    def net_proc_backward_with_data(self, filter_idx, data, backprop_layer):

        diffs = self.net.blobs[backprop_layer].diff * 0
        if self.ds.backprop_xy == 'sin':
            x,y = np.unravel_index(np.argmax(data[filter_idx]), data[filter_idx].shape)
            diffs[0][filter_idx][x][y] = data[filter_idx][x][y]
        elif self.ds.backprop_xy == 'all':
            diffs[0][filter_idx] = data[filter_idx]
        assert self.back_mode in ('grad', 'deconv')
        if self.back_mode == 'grad':
            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
        else:
            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)

    def get_orig_xy(self, xy, resize_ratio):
        if np.isnan(xy[0]) or np.isnan(xy[1]):
            return (float('nan'),float('nan'))

        new_x = int(round(xy[0]*resize_ratio))
        new_y = int(round(xy[1]*resize_ratio))
        return (new_x, new_y)

    def get_frame_xyz(self, data, frame_name):
        return data.pose_dict[frame_name][0]

    def gen_receptive_grid(self, receptive_field_size):
        return np.mgrid[0:receptive_field_size,0:receptive_field_size]

    def get_closest_xyz_from_point_cloud(self, pc, xy, max_width):
        return closest_pc_value_fast(pc,xy[0],xy[1],max_width)

    def get_average_xyz_from_point_cloud(self, pc, xy, receptive_grid):
        if np.isnan(xy[0]):
            return [float('nan'),float('nan'),float('nan')]
            print "filter response zero no max xy", xy

        pc_array = pc.reshape((pc.shape[0]*pc.shape[1], pc.shape[2]))

        # receptive grid has shape (2,w,w) that contains the grid x idx and y idx
        grid = np.zeros(receptive_grid.shape)
        grid[0] = xy[0] + receptive_grid[0]
        grid[1] = xy[1] + receptive_grid[1]

        # this step flattens to 2 arrays of x coordinates and y coordinates
        xy_receptive_list =np.reshape(grid, [2,-1])

        # remove out of bound index
        xy_receptive_list_filtered  = np.array([]).reshape([2,0])

        for i in range(xy_receptive_list.shape[1]):
            x = xy_receptive_list[0,i]
            y = xy_receptive_list[1,i]
            if x < pc.shape[0] and x >= 0 and y < pc.shape[1] and y >= 0:
                xy_receptive_list_filtered = np.append(xy_receptive_list_filtered, xy_receptive_list[:,i].reshape([2,1]), axis=1)

        idx_receptive_list = np.ravel_multi_index(xy_receptive_list_filtered.astype(int),pc.shape[0:2])
        avg = np.nanmean(pc_array[idx_receptive_list],axis=0)
        if np.isnan(avg[0]) or np.isnan(avg[1]) or np.isnan(avg[2]):
            print "nan found", xy

        return avg.tolist()


    def mask_out(self, data, mask):
        # print "data shape", data.shape
        dim = data.shape

        for y in range(dim[2]):
            for x in range(dim[3]):
                if is_masked((dim[2],dim[3]),(x,y),mask):
                    data[:,:,y,x] = 0

        return data


    def load_layer(self, data_list, layer):

        layer_list = np.array([]).reshape([0] + list(self.net.blobs[layer].data.shape[1:]))

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()
            # print "img", img.shape
            self.net_proc_forward_layer(data.img, data.mask)
            # self.net_preproc_forward(img)
            layer_list = np.append(layer_list, self.net.blobs[layer].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return layer_list


    def load_layer_fix_filter_list(self, load_layer, fix_layer, fix_layer_data_list, data_list, filter_idx):

        layer_diff_list = np.zeros([len(data_list)] + list(self.net.blobs[load_layer].diff.shape[1:]))
        img_src_list = np.zeros([len(data_list)] + list(self.net.blobs['data'].data.shape[2:]))

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()
            self.net_proc_forward_layer(data.img, data.mask)
            layer_diff_list[idx,:], img_src_list[idx,:], _ = self.load_layer_fix_filter(load_layer, fix_layer, fix_layer_data_list[idx], data, filter_idx)

        return layer_diff_list, img_src_list

    # perform forward path and backward path while zeroing out all filter response except for filter_idx
    # return layer_diff_list which is the load layer diff and img_src_list which is the abs diff if back propagate to image layer
    def load_layer_fix_filter(self, load_layer, fix_layer, fix_layer_diff, data, filter_idx):
        #self.net_proc_forward_layer(data.img, data.mask)
        self.net_proc_backward_with_data(filter_idx, fix_layer_diff, fix_layer)
        layer_diff = self.net.blobs[load_layer].diff[0,:]
        img_src_color = self.net.blobs['data'].diff[0,:].transpose((1,2,0))
        # mean is to average over all filters (RGB)
        if self.ds.img_src_loc == "absolute":
            img_src = np.absolute(self.net.blobs['data'].diff[0,:]).mean(axis=0)
        elif self.ds.img_src_loc == "relu":
            img_src = np.maximum(self.net.blobs['data'].diff[0,:],0).mean(axis=0)
        elif self.ds.img_src_loc == "norm":
            img_src = np.linalg.norm(self.net.blobs['data'].diff[0,:], axis=0)
        # make a copy
        layer_diff = copy.deepcopy(layer_diff)
        img_src = copy.deepcopy(img_src)
        img_src_color = copy.deepcopy(img_src_color)

        return layer_diff, img_src, img_src_color

    # binaraizes such that output is a 1-d array where each entry is whether a filter fires, also ouputs the max value
    def binarize(self, data, threshold):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        max_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            max_value = np.amax(filter)
            if max_value > threshold:
                bin_data[id] = 1
            max_data[id] = max(0.,max_value)
        return bin_data, max_data
