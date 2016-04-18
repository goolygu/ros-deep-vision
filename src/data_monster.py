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
from numpy_cache import FIFOLimitedArrayCache
from app_base import BaseApp
# from core import CodependentThread
from image_misc import *

from time import gmtime, strftime
from utils import DescriptorHandler, Descriptor
import settings
import caffe

from collections import namedtuple
import yaml
from data_collector import Data
from data_ploter import *
from distribution import *
import pcl

from perception_msgs.srv import String2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import data_list as dl
import copy
import scipy
from scipy.special import expit

from data_util import *

from data_settings import *

class DataMonster:

    def __init__(self, settings, data_settings):
        print 'initialize'
        self.settings = settings
        self.ds = data_settings
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
        self.threshold = {}
        self.threshold['conv5'] = self.ds.thres_conv5
        self.threshold['conv4'] = self.ds.thres_conv4#10
        self.threshold['conv3'] = self.ds.thres_conv3#2
        self.threshold['conv2'] = 0
        self.threshold['conv1'] = 0
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=100)


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
        self.back_mode = self.ds.back_prop_mode
        w = self.ds.avg_pointcloud_width
        self.average_grid = np.mgrid[-w:w,-w:w]
        self.visualize = True
        self.show_backprop = True
        # self.data_file_list = []

        # self.descriptor_handler = DescriptorHandler(self.settings.ros_dir + '/eric_data_set/model_seg/', self.descriptor_layers)

    def set_path(self, path):
        self.path = path

    def train_each_case(self):
        data_dic = get_data_dic(self.path, dl.data_name_list)
        dist_dic = {}
        for case in data_dic:
            print "train case", case
            dist_dic[case] = self.train(data_dic[case])
        return dist_dic

    def train_all(self):
        data_list = get_data_all(self.path)
        return self.train(data_list)

    def train(self, data_list):
        distribution = Distribution()

        # data_list = data_list[0:26]

        img_list, mask_list = self.load_img_mask(data_list, self.path)

        conv5_list = self.load_conv5(img_list, mask_list)

        # print "conv5", conv5_list.shape

        filter_idx_list_conv5 = self.find_consistent_filters(conv5_list, self.threshold["conv5"], self.ds.n_conv5_f)#[87]#self.find_consistent_filters(conv5_list, self.threshold["conv5"], 3)
        print "consistent filter conv5", filter_idx_list_conv5
        # filter_idx_list_conv5 = filter_idx_list_conv5[0:3]

        # frame_list_conv5 = ["r2/left_palm"]
        # dist_list = self.gen_distribution(filter_idx_list_conv5, data_list, conv5_list, frame_list_conv5)
        # self.set_distribution(distribution, frame_list_conv5, filter_idx_list_conv5, dist_list, 'conv5')
        distribution.set_tree_list([],filter_idx_list_conv5)
        # handle conv 4 filters
        for filter_idx_5 in filter_idx_list_conv5:

            print "handling filter", filter_idx_5, "layer conv5"
            conv4_list, bp_layers_5 = self.load_layer_fix_filter("conv4", "conv5", conv5_list, img_list, mask_list, filter_idx_5)
            filter_idx_list_conv4 = self.find_consistent_filters(conv4_list, self.threshold["conv4"], self.ds.n_conv4_f)#[190,133]#
            if (len(filter_idx_list_conv4) == 0):
                continue
            print "consistent filter conv4", filter_idx_list_conv4
            # filter_idx_list_conv4 = filter_idx_list_conv4[0:3]

            # filter_idx_list_conv3_dict = {}
            # filter_idx_list_conv3_dict[190] = [168,51,181,70,188,168,154,157,19,17]
            # filter_idx_list_conv3_dict[133] = [18,157,54,37,97,157,59,171,11]

            bp_list_4 = np.zeros([len(data_list)] + [len(filter_idx_list_conv4)] + list(self.net.blobs["data"].diff.shape[2:]))
            distribution.set_tree_list([filter_idx_5],filter_idx_list_conv4)

            for i, filter_idx_4 in enumerate(filter_idx_list_conv4):
                print "handling filter", filter_idx_4, "layer conv4"
                conv3_list, bp_layers_4 = self.load_layer_fix_filter("conv3", "conv4", conv4_list, img_list, mask_list, filter_idx_4)
                bp_list_4[:,i,:] = bp_layers_4
                filter_idx_list_conv3 = self.find_consistent_filters(conv3_list, self.threshold["conv3"], self.ds.n_conv3_f)#filter_idx_list_conv3_dict[filter_idx_4]#
                if (len(filter_idx_list_conv3) == 0):
                    continue
                print "consistent filter conv3", filter_idx_list_conv3
                # filter_idx_list_conv3 = filter_idx_list_conv3[0:3]

                bp_list_3 = np.zeros([len(data_list)] + [len(filter_idx_list_conv3)] + list(self.net.blobs["data"].diff.shape[2:]))
                distribution.set_tree_list([filter_idx_5, filter_idx_4],filter_idx_list_conv3)

                for j, filter_idx_3 in enumerate(filter_idx_list_conv3):
                    print "handling filter", filter_idx_3, "layer conv3"
                    conv2_list, bp_layers_3 = self.load_layer_fix_filter("conv2", "conv3", conv3_list, img_list, mask_list, filter_idx_3)
                    bp_list_3[:, j,:] = bp_layers_3

                # gen distribution
                # bp_list_3 = np.swapaxes(bp_list_3, 0, 1)

                # frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]
                dist_list_3 = self.gen_distribution_bp(filter_idx_list_conv3, data_list, conv3_list, bp_list_3, self.ds.frame_list_conv3, self.threshold["conv3"], 'conv3')
                self.set_distribution(distribution, self.ds.frame_list_conv3, filter_idx_list_conv3, dist_list_3, [filter_idx_5, filter_idx_4])



            # bp_list_4 = np.swapaxes(bp_list_4, 0, 1)

            # frame_list_conv4 = ["r2/left_palm"]#["r2/left_thumb_tip","r2/left_index_tip"]
            dist_list_4 = self.gen_distribution_bp(filter_idx_list_conv4, data_list, conv4_list, bp_list_4, self.ds.frame_list_conv4, self.threshold["conv4"], 'conv4')
            self.set_distribution(distribution, self.ds.frame_list_conv4, filter_idx_list_conv4, dist_list_4, [filter_idx_5])

            # self.filter_distribution(distribution, 0.03**2)


        return distribution

    def cross_validation(self, path, name, train):
        self.visualize = False
        evaluate_on_full_dist = False

        data_name_dic = get_data_name_dic(self.path, dl.data_name_list)
        dist_dic = {}
        result_xyz = {}
        result = {}
        fail_count = {}

        if evaluate_on_full_dist:
            full_distribution = {}

            for case in data_name_dic:
                print "full training", case
                train_data_list_full = []
                for j, train_object in enumerate(data_name_dic[case]):
                    train_data_list_full = train_data_list_full + data_name_dic[case][train_object]

                full_distribution[case] = self.train(train_data_list_full)
                full_distribution[case].save(path + '/data/', "[" + case + "]" + name)

        train_data_list = {}
        test_data_list = {}

        # create train data list and test data list
        for k, case in enumerate(data_name_dic):
            # print "train case", case

            train_data_list[case] = {}
            test_data_list[case] = {}
            for i, test_object in enumerate(data_name_dic[case]):
                # print "leave", i, test_object, "out"
                train_data_list[case][test_object] = []
                test_data_list[case][test_object] = []
                for j, train_object in enumerate(data_name_dic[case]):
                    if j != i:
                        train_data_list[case][test_object] += data_name_dic[case][train_object]
                    else:
                        test_data_list[case][test_object] += data_name_dic[case][train_object]
                # print "train_data_list", train_data_list

        # train and save
        if train:
            for k, case in enumerate(data_name_dic):
                print "train case", case
                for i, test_object in enumerate(data_name_dic[case]):
                    print "leave", i, test_object, "out"

                    distribution = self.train(train_data_list[case][test_object])

                    if evaluate_on_full_dist:
                        for other_case in full_distribution:
                            if other_case != case:
                                distribution.merge(full_distribution[other_case])
                    # print "test_data_list", test_data_list

                    distribution.save_exact(path + '/data/distribution/cross_validation/', "[" + case + "][leave_" + test_object + "]" + name)

        # test
        for k, case in enumerate(data_name_dic):
            result[case] = {}
            result_xyz[case] = {}
            fail_count[case] = {}
            print "test case", case
            for i, test_object in enumerate(data_name_dic[case]):
                print "test", i, test_object
                distribution = Distribution()
                distribution.load_exact(path + '/data/distribution/cross_validation/', "[" + case + "][leave_" + test_object + "]" + name)
                result_xyz[case][test_object], result[case][test_object], fail_count[case][test_object] = self.test_accuracy(distribution, test_data_list[case][test_object])

        accuracy_sum = 0.0
        count = 0.0
        for case in result:
            for test_object in result[case]:
                for frame in result[case][test_object]:
                    accuracy_sum += result[case][test_object][frame]
                    count += 1
                    print case, test_object, frame, result[case][test_object][frame]

        print "average", accuracy_sum/count

        test_name = self.ds.get_test_name()


        with open(self.path + "/result/cross_validation_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

        with open(self.path + "/result/cross_validation_xyz_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(result_xyz, f, default_flow_style=False)

        with open(self.path + "/result/cross_validation_fail_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(fail_count, f, default_flow_style=False)


    def test_accuracy(self, distribution, data_list):

        img_list, mask_list = self.load_img_mask(data_list, self.path)

        diff_sum_dic = {}
        diff_count = {}
        diff_fail = {}

        for idx, data in enumerate(data_list):
            print data.name

            filter_xyz_dict = self.get_all_filter_xyz(data, distribution, img_list[idx], mask_list[idx])

            distribution_cf = self.get_distribution_cameraframe(distribution, filter_xyz_dict)

            avg_dic = self.model_distribution(distribution_cf)

            if self.visualize:
                self.show_point_cloud(data.name)
                self.show_distribution(distribution_cf)
            # print "avg", avg_dic
            for frame in avg_dic:
                # get ground truth frame location
                frame_xyz = np.array(self.get_frame_xyz(data, frame))
                if not frame in diff_sum_dic:
                    diff_sum_dic[frame] = np.array([0.,0.,0.])
                    diff_count[frame] = 0
                    diff_fail[frame] = 0
                # print "frame_xyz", frame_xyz
                if not np.isnan(avg_dic[frame][0]):
                    diff_sum_dic[frame] += np.absolute(frame_xyz - avg_dic[frame])
                    diff_count[frame] += 1
                else:
                    diff_fail[frame] += 1

        diff_avg_dic = {}
        diff_dist_dic = {}
        for frame in diff_sum_dic:
            avg_xyz = diff_sum_dic[frame]/diff_count[frame]
            diff_avg_dic[frame] = avg_xyz.tolist()
            diff_dist_dic[frame] = (np.linalg.norm(avg_xyz)).tolist()

            # print frame, diff_avg_dic[frame]

        return diff_avg_dic, diff_dist_dic, diff_fail

            # self.show_distribution(distribution_cf)

    def test(self, distribution):

        data_list = [get_data_by_name(self.path,dl.data_name_list[15])]
        img_list, mask_list = self.load_img_mask(data_list, self.path)

        for idx, data in enumerate(data_list):
            print data.name
            # filter_xyz_dict = self.get_all_filter_xyz_test(data, distribution, img_list[idx], mask_list[idx])
            # while not rospy.is_shutdown():
            #     pass
            filter_xyz_dict = self.get_all_filter_xyz(data, distribution, img_list[idx], mask_list[idx])
            # self.show_feature(filter_xyz_dict)
            # print filter_xyz_dict
            distribution_cf = self.get_distribution_cameraframe(distribution, filter_xyz_dict)
            self.show_point_cloud(data.name)
            self.model_distribution(distribution_cf)
            self.show_distribution(distribution_cf)


    def filter_distribution(self, dist, threshold):
        new_dist = Distribution()

        data_dict = dist.data_dict
        for sig in data_dict:
            for frame in data_dict[sig]:
                point_list = data_dict[sig][frame]
                var = np.nanvar(point_list, axis = 0)
                if np.sum(var) < threshold:
                    new_dist.set(sig, frame, point_list)
        new_dist.filter_tree = dist.filter_tree
        return new_dist


    def model_distribution(self, dist_cf):
        dist_list = {}
        dist_list["r2/left_palm"] = np.array([]).reshape([0,3])
        dist_list["r2/left_thumb_tip"] = np.array([]).reshape([0,3])
        dist_list["r2/left_index_tip"] = np.array([]).reshape([0,3])

        # concatenate all points in camera frame of same robot joint
        for sig in dist_cf:
            for frame in dist_cf[sig]:
                # print type(dist_cf[sig][frame])
                dist_list[frame] = np.concatenate((dist_list[frame], dist_cf[sig][frame]), axis=0)

        # print dist_list

        avg_dic = {}
        cal_mean = True
        for frame in dist_list:
            if cal_mean:
                avg_dic[frame] = np.nanmean(dist_list[frame], axis=0)
            else:
                avg_dic[frame] = find_max_density(dist_list[frame])

        color_map = {}
        color_map["r2/left_palm"] = (0.5,0,0)
        color_map["r2/left_thumb_tip"] = (0,0.5,0)
        color_map["r2/left_index_tip"] = (0,0,0.5)
        count = 0

        if self.visualize:
            while not rospy.is_shutdown() and count < 10000:
                for frame in avg_dic:
                    ns = frame
                    self.publish_sphere_list([avg_dic[frame]], color_map[frame], 0, ns)
                count += 1

        return avg_dic


    def set_distribution(self, distribution, frame_list, filter_idx_list, dist_list, parent_filters):
        for j, frame in enumerate(frame_list):
            for i, filter_idx in enumerate(filter_idx_list):
                distribution.set(parent_filters + [filter_idx], frame, dist_list[j,i,:,:])

    def show_distribution(self, dist_cf):
        color_map = {}
        color_map["r2/left_palm"] = (1,0,0)
        color_map["r2/left_thumb_tip"] = (0,1,0)
        color_map["r2/left_index_tip"] = (0,0,1)

        count = 0
        while not rospy.is_shutdown() and count < 10000:
            idx = 0
            for sig in dist_cf:
                for frame in dist_cf[sig]:
                    ns = "/" + "/".join([str(c) for c in sig]) + "-" + frame
                    self.publish_point_list(dist_cf[sig][frame], color_map[frame], idx, ns)
                    # idx += 1
                count += 1

    def show_feature(self, filter_xyz_dict):

        for sig in filter_xyz_dict:
            print sig, filter_xyz_dict[sig]

        while not rospy.is_shutdown():
            idx = 0
            for sig in filter_xyz_dict:
                ns = "/" + "/".join([str(c) for c in sig])

                self.publish_point_list([filter_xyz_dict[sig]], (1,0,0), idx, ns)
                    # idx += 1


    def get_distribution_cameraframe(self, dist, filter_xyz_dict):
        dist_cf = copy.deepcopy(dist.data_dict)
        for sig in dist.data_dict:
            for frame in dist.data_dict[sig]:
                if sig in filter_xyz_dict:
                    dist_cf[sig][frame] += filter_xyz_dict[sig]
                else:
                    if sig in dist_cf:
                        dist_cf.pop(sig)
        return dist_cf

    # def fill_xyz_dict(self, data, layer_data, threshold, xyz_dict, parent_filters):
    #     max_xy_list = []
    #     resize_ratio = get_after_crop_size()[0] / layer_data.shape[2]
    #     receptive_field_size = int(round(resize_ratio))
    #     receptive_grid = self.gen_receptive_grid(receptive_field_size)
    #
    #     for filter_idx in distribution.data_dict[layer]:
    #         max_xy = self.get_filter_max_xy(layer_data[0,filter_idx], threshold)
    #         orig_xy = self.get_orig_xy(max_xy, resize_ratio)
    #         max_xy_list.append(orig_xy)
    #     max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, data.name, max_xy_list, receptive_grid)
    #
    #     for i, filter_idx in enumerate(distribution.data_dict[layer]):
    #         xyz = max_xyz_list[i]
    #         xyz_dict[parent_filters + [filter_idx]] = xyz

    def get_max_xyz(self, filter_data, pc_array, threshold):
        resize_ratio = get_after_crop_size()[0] / filter_data.shape[0]
        receptive_field_size = int(round(resize_ratio))
        receptive_grid = self.gen_receptive_grid(receptive_field_size)

        max_xy = self.get_filter_max_xy(filter_data, threshold)
        orig_xy = self.get_orig_xy(max_xy, resize_ratio)
        max_xyz = self.get_average_xyz_from_point_cloud_array(pc_array, [orig_xy], receptive_grid)
        return max_xyz[0]

    def get_avg_xyz(self, layer_data, pc_array, threshold):
        resize_ratio = get_after_crop_size()[0] / layer_data.shape[0]

        avg_xy = self.get_filter_avg_xy(layer_data, threshold)
        orig_xy = self.get_orig_xy(avg_xy, resize_ratio)
        avg_xyz = self.get_average_xyz_from_point_cloud_array(pc_array, [orig_xy], self.average_grid)
        return avg_xyz[0], avg_xy


    def get_all_filter_xyz(self, data, dist, img, mask):

        pc_array = self.get_point_cloud_array(self.path, data.name)
        xyz_dict = {}
        self.net_proc_forward_layer(img, mask)
        conv5_data = copy.deepcopy(self.net.blobs['conv5'].data)

        filter_idx_5_list = []
        if self.ds.filter_test == 'top':
            filter_idx_5_list = self.get_top_filters_in_list(conv5_data[0], dist.filter_tree, self.ds.conv5_top)
        elif self.ds.filter_test == 'all':
            filter_idx_5_list = dist.filter_tree

        for filter_idx_5 in filter_idx_5_list:
            print filter_idx_5
            layer = 'conv5'

            if not self.filter_response_pass_threshold(conv5_data[0,filter_idx_5], self.ds.thres_conv5_test):
                continue

            self.net_proc_forward_layer(img, mask)
            self.net_proc_backward_with_data(filter_idx_5, conv5_data[0], layer)
            bp_5 = copy.deepcopy(self.net.blobs['data'].diff)
            xyz_dict[(filter_idx_5,)], max_xy = self.get_avg_xyz(np.absolute(bp_5[0].mean(axis=0)), pc_array, 0)

            self.show_gradient(str((filter_idx_5)), self.net.blobs['data'].diff, max_xy, 0)

            conv4_data = copy.deepcopy(self.net.blobs['conv4'].diff)


            filter_idx_4_list = []
            if self.ds.filter_test == 'top':
                filter_idx_4_list = self.get_top_filters_in_list(conv4_data[0], dist.filter_tree[filter_idx_5], self.ds.conv4_top)
            elif self.ds.filter_test == 'all':
                filter_idx_4_list = dist.filter_tree[filter_idx_5]

            for filter_idx_4 in filter_idx_4_list:
                print filter_idx_5, filter_idx_4
                layer = 'conv4'

                if not self.filter_response_pass_threshold(conv4_data[0,filter_idx_4], self.ds.thres_conv4_test):
                    continue

                self.net_proc_forward_layer(img, mask)
                self.net_proc_backward_with_data(filter_idx_4, conv4_data[0], layer)

                bp_4 = copy.deepcopy(self.net.blobs['data'].diff)
                xyz_dict[(filter_idx_5, filter_idx_4)], max_xy = self.get_avg_xyz(np.absolute(bp_4[0].mean(axis=0)), pc_array, 0)

                self.show_gradient(str((filter_idx_5, filter_idx_4)), self.net.blobs['data'].diff, max_xy, 0)

                conv3_data = copy.deepcopy(self.net.blobs['conv3'].diff)

                filter_idx_3_list = []
                if self.ds.filter_test == 'top':
                    filter_idx_3_list = self.get_top_filters_in_list(conv3_data[0], dist.filter_tree[filter_idx_5][filter_idx_4], self.ds.conv3_top)
                elif self.ds.filter_test == 'all':
                    filter_idx_3_list = dist.filter_tree[filter_idx_5][filter_idx_4]

                for filter_idx_3 in filter_idx_3_list:
                    print filter_idx_5, filter_idx_4, filter_idx_3
                    layer = 'conv3'

                    if not self.filter_response_pass_threshold(conv3_data[0,filter_idx_3], self.ds.thres_conv3_test):
                        continue

                    self.net_proc_forward_layer(img, mask)
                    self.net_proc_backward_with_data(filter_idx_3, conv3_data[0], layer)

                    bp_3 = copy.deepcopy(self.net.blobs['data'].diff)
                    xyz_dict[(filter_idx_5, filter_idx_4, filter_idx_3)], max_xy = self.get_avg_xyz(np.absolute(bp_3[0].mean(axis=0)), pc_array, 0)

                    self.show_gradient(str((filter_idx_5, filter_idx_4, filter_idx_3)), self.net.blobs['data'].diff, max_xy, 0)

        return xyz_dict

    def show_gradient(self, name, grad_blob, xy_dot=(0,0), threshold=0):
        if not self.visualize or not self.show_backprop:
            return
        grad_blob = grad_blob[0]                    # bc01 -> c01
        grad_blob = grad_blob.transpose((1,2,0))    # c01 -> 01c
        grad_img = grad_blob[:, :, (2,1,0)]  # e.g. BGR -> RGB

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

        back_filt_mode = 'raw'
        if back_filt_mode == 'raw':
            grad_img = norm01c(grad_img, 0)
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
            for i in range(-8,8):
                for j in range(-8,8):
                    grad_img[i+xy_dot[0],j+xy_dot[1]] = [1,0,0]


        # if not np.isnan(xy_dot2[0]) and not np.isnan(xy_dot2[1]):
        #     for i in range(-3,3):
        #         for j in range(-3,3):
        #             grad_img[i+xy_dot2[0],j+xy_dot2[1]] = [1,0,1]

        cv2.imshow(name, grad_img)
        cv2.waitKey(100)

    def find_consistent_filters(self, conv_list, threshold, number):
        hist = np.zeros(conv_list.shape[1])
        max_hist = np.zeros(conv_list.shape[1])
        max_sum = np.zeros(conv_list.shape[1])
        for idx, conv in enumerate(conv_list):
            # print idx

            bin_data, max_data = self.binarize(conv, threshold)
            hist = hist + bin_data
            max_hist = np.amax(np.concatenate((max_hist[np.newaxis,...],max_data[np.newaxis,...]),axis=0),axis=0)
            max_data = scipy.special.expit(max_data)
            max_sum = max_data + max_sum

        # print "hist", hist
        # print "max hist", max_hist
        if self.ds.top_filter == 'above':
            filter_idx_list = np.argsort(hist)[::-1]
        elif self.ds.top_filter == 'max':
            filter_idx_list = np.argsort(max_sum)[::-1]

        # print "top filters counts", hist[filter_idx_list[0:number+10]]
        # print "top filters", filter_idx_list[0:number+10]
        # print "max value", max_hist[filter_idx_list[0:number+10]]

        for i in range(number+1):
            if hist[filter_idx_list[i]] == 0:
                number = i
                # print "break", number
                break
        return filter_idx_list[0:number]

    def get_filter_avg_xy(self, filter_response, threshold):
        # print "max", np.amax(filter_response)
        assert filter_response.ndim == 2, "filter size incorrect"
        xy_grid = np.mgrid[0:filter_response.shape[0], 0:filter_response.shape[0]]

        if np.sum(filter_response) == 0:
            return np.array([float('nan'),float('nan')])

        filter_response_norm = filter_response / float(np.sum(filter_response))
        avg_x = np.sum(xy_grid[0] * filter_response_norm)
        avg_y = np.sum(xy_grid[1] * filter_response_norm)

        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])

        return np.around(np.array([avg_x, avg_y])).astype(int)


    def get_filter_max_xy(self, filter_response, threshold):
        assert filter_response.ndim == 2, "filter size incorrect"
        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])
        max_idx = np.argmax(filter_response, axis=None)
        max_xy = np.unravel_index(max_idx, filter_response.shape)
        return max_xy

    def filter_response_pass_threshold(self, filter_response, threshold):
        max_v = np.nanmax(filter_response)
        # print "max", max_v
        if max_v <= threshold:
            print "failed threshold", max_v
            return False
        else:
            return True

    def get_top_filters_in_list(self, layer_response, filter_tree, number):

        filter_list = list(filter_tree)
        max_list = np.zeros(len(filter_list))
        for i, filter_id in enumerate(filter_list):
            max_list[i] = np.amax(layer_response[filter_id])

        sorted_idx_list = np.argsort(max_list)[::-1]
        sorted_idx_list = sorted_idx_list[0:number]

        sorted_filter_idx_list = []
        for idx in sorted_idx_list:
            sorted_filter_idx_list.append(filter_list[idx])

        return sorted_filter_idx_list



    # returns a distribution list of shape(num_frames, number of filters, num of data, 3)
    # distribution contains diff of frame xyz to feature xyz
    def gen_distribution_bp(self, filter_idx_list, data_list, conv_list, bp_list, frame_list, threshold, layer):

        dist_list = np.empty([len(frame_list),len(filter_idx_list),len(data_list),3])
        # 280/13
        img_size = get_after_crop_size()
        # resize_ratio = img_size[0] / bp_list.shape[2]
        resize_ratio = img_size[0] / bp_list.shape[2]

        for idx, bp in enumerate(bp_list):
            print idx,
            sys.stdout.flush()
            data = data_list[idx]
            name = data.name
            max_idx_list = []
            max_xy_list = []
            for i, filter_idx in enumerate(filter_idx_list):
                # cv2.imshow(layer + " " + str(idx)+" "+ str(filter_idx),norm01c(bp[i], 0))
                # cv2.waitKey(200)
                if not self.filter_response_pass_threshold(conv_list[idx][filter_idx], threshold):
                    max_xy_list.append((float('nan'),float('nan')))
                    continue
                max_xy = self.get_filter_avg_xy(bp[i], 0)
                # max_xy = self.get_bp_avg_xy(bp_list[idx])
                orig_xy = self.get_orig_xy(max_xy, resize_ratio)

                # print "orig", orig_xy
                max_xy_list.append(orig_xy)

            max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, name, max_xy_list, self.average_grid)
            # max_xyz_list = self.get_xyz_from_point_cloud(dir, name, max_idx_list)
            for frame_idx, frame in enumerate(frame_list):
                frame_xyz = np.array(self.get_frame_xyz(data, frame))
                diff_list = [frame_xyz - feature_xyz for feature_xyz in max_xyz_list]
                dist_list[frame_idx,:,idx,:] = np.array(diff_list)

        # for frame_idx, frame in enumerate(frame_list):
        #     for i, filter_idx in enumerate(filter_idx_list):
        #         plot_dist(dist_list[frame_idx,i,:,:])

        return dist_list

    def gen_distribution(self, filter_idx_list, data_list, conv_list, frame_list, threshold):

        dist_list = np.empty([len(frame_list),len(filter_idx_list),len(data_list),3])
        # 280/13
        img_size = get_after_crop_size()
        # resize_ratio = img_size[0] / bp_list.shape[2]
        resize_ratio = img_size[0] / conv_list.shape[2]

        receptive_field_size = int(round(resize_ratio))
        receptive_grid = self.gen_receptive_grid(receptive_field_size)

        for idx, conv in enumerate(conv_list):
            print idx,
            sys.stdout.flush()
            data = data_list[idx]
            name = data.name
            max_idx_list = []
            max_xy_list = []
            for filter_idx in filter_idx_list:
                max_xy = self.get_filter_max_xy(conv[filter_idx], threshold)
                # max_xy = self.get_bp_avg_xy(bp_list[idx])
                orig_xy = self.get_orig_xy(max_xy, resize_ratio)

                # print "orig", orig_xy
                max_xy_list.append(orig_xy)

            max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, name, max_xy_list, receptive_grid)
            # max_xyz_list = self.get_xyz_from_point_cloud(dir, name, max_idx_list)
            for frame_idx, frame in enumerate(frame_list):
                frame_xyz = np.array(self.get_frame_xyz(data, frame))
                diff_list = [frame_xyz - feature_xyz for feature_xyz in max_xyz_list]
                dist_list[frame_idx,:,idx,:] = np.array(diff_list)

        # for frame_idx, frame in enumerate(frame_list):
        #     for i, filter_idx in enumerate(filter_idx_list):
        #         plot_dist(dist_list[frame_idx,i,:,:])

        return dist_list


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
        # cv2.imshow("blob", np.swapaxes(data_blob[0],0,2))
        # cv2.waitKey(200)
        # self.mask_out(data_blob, mask)
        # cv2.imshow("blobmasked", data_blob[0])
        # cv2.waitKey(200)
        mode = 2
        if mode == 0:
            self.net.blobs['data'].data[...] = data_blob
            self.net.forward_from_to(start='conv1',end='relu1')
            self.mask_out(self.net.blobs['conv1'].data, mask)
            self.net.forward_from_to(start='relu1',end='prob')
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
            self.net.forward_from_to(start='relu5',end='prob')
        elif mode == 2:
            for idx in range(len(self.available_layer)-1):
                output = self.net.forward(data=data_blob,start=self.available_layer[idx],end=self.available_layer[idx+1])
                if self.available_layer[idx].startswith("conv"):
                    self.mask_out(self.net.blobs[self.available_layer[idx]].data, mask)


    def net_proc_backward(self, filter_idx, backprop_layer):

        diffs = self.net.blobs[backprop_layer].diff * 0
        diffs[0][filter_idx] = self.net.blobs[backprop_layer].data[0,filter_idx]

        assert self.back_mode in ('grad', 'deconv')
        if self.back_mode == 'grad':
            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
        else:
            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)

    def net_proc_backward_with_data(self, filter_idx, data, backprop_layer):

        diffs = self.net.blobs[backprop_layer].diff * 0
        if self.ds.backprop_xy == 'sin':
            x,y = np.unravel_index(np.argmax(data[filter_idx]), data[filter_idx].shape)
            diffs[0][filter_idx][x][y] = data[filter_idx][x][y]
        else:
            diffs[0][filter_idx] = data[filter_idx]
        assert self.back_mode in ('grad', 'deconv')
        if self.back_mode == 'grad':
            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
        else:
            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)

    def get_orig_xy(self, xy, resize_ratio):
        if np.isnan(xy[0]) or np.isnan(xy[1]):
            return (float('nan'),float('nan'))

        new_x = int(round(xy[0]*resize_ratio + self.xy_bias[0]))
        new_y = int(round(xy[1]*resize_ratio + self.xy_bias[1]))
        return (new_x, new_y)

    def get_frame_xyz(self, data, frame_name):
        return data.pose_dict[frame_name][0]

    def gen_receptive_grid(self, receptive_field_size):
        return np.mgrid[0:receptive_field_size,0:receptive_field_size]
        # x = np.arange(0,receptive_field_size)
        # y = np.arange(0,receptive_field_size)
        # xv, yv = np.meshgrid(x,y)
        # return (xv, yv)

    def get_point_cloud_array(self, path, name):
        p = pcl.PointCloud()
        p.from_file(path + name + "_seg.pcd")
        a = np.asarray(p)
        return a

    def get_average_xyz_from_point_cloud(self, path, name, max_xy_list, receptive_grid):
        pc_array = self.get_point_cloud_array(path, name)
        return self.get_average_xyz_from_point_cloud_array(pc_array, max_xy_list, receptive_grid)

    def get_average_xyz_from_point_cloud_array(self, pc_array, max_xy_list, receptive_grid):

        output = []
        for xy in max_xy_list:
            if np.isnan(xy[0]):
                output.append([float('nan'),float('nan'),float('nan')])
                print "filter response zero no max xy", xy
                continue
            grid = np.zeros(receptive_grid.shape)
            grid[0] = xy[0] + receptive_grid[0]
            grid[1] = xy[1] + receptive_grid[1]
            xy_receptive_list =np.reshape(grid, [2,-1])
            idx_receptive_list = np.ravel_multi_index(xy_receptive_list.astype(int),(480,640))
            avg = np.nanmean(pc_array[idx_receptive_list],axis=0)
            if np.isnan(avg[0]) or np.isnan(avg[1]) or np.isnan(avg[2]):
                print "nan found", xy
            # avg = np.nanmedian(a[idx_receptive_list],axis=0)
            output.append(avg)

        return np.array(output)

    def show_point_cloud(self, name):
        rospy.wait_for_service('show_point_cloud')
        try:
            show_point_cloud = rospy.ServiceProxy('show_point_cloud', String2)
            resp = show_point_cloud(name,'')
            return resp.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def array_to_pose_msg(self, point_list):
        msg_list = []
        for point in point_list:
            if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
                continue
            p_msg = Point()
            p_msg.x = point[0]
            p_msg.y = point[1]
            p_msg.z = point[2]
            msg_list.append(p_msg)
        return tuple(msg_list)

    def publish_point_list(self, point_list, color, idx, ns):

        pl_marker = Marker()
        pl_marker.header.frame_id = "/r2/head/asus_depth_optical_frame"
        pl_marker.header.stamp = rospy.Time()
        pl_marker.id = idx
        pl_marker.ns = ns
        pl_marker.type = Marker.POINTS
        # pl_marker.pose.position.x = 1
        # pl_marker.pose.position.y = 1
        # pl_marker.pose.position.z = 1
        pl_marker.scale.x = 0.01
        pl_marker.scale.y = 0.01
        pl_marker.scale.z = 0.01
        pl_marker.color.a = 0.4
        pl_marker.color.r = color[0]
        pl_marker.color.g = color[1]
        pl_marker.color.b = color[2]
        pl_marker.lifetime = rospy.Duration.from_sec(1200)
        pl_marker.action = Marker.ADD
        pl_marker.points = self.array_to_pose_msg(point_list)

        self.marker_pub.publish(pl_marker)

    def publish_sphere_list(self, point_list, color, idx, ns):

        pl_marker = Marker()
        pl_marker.header.frame_id = "/r2/head/asus_depth_optical_frame"
        pl_marker.header.stamp = rospy.Time()
        pl_marker.id = idx
        pl_marker.ns = ns
        pl_marker.type = Marker.SPHERE_LIST
        # pl_marker.pose.position.x = 1
        # pl_marker.pose.position.y = 1
        # pl_marker.pose.position.z = 1
        pl_marker.scale.x = 0.04
        pl_marker.scale.y = 0.04
        pl_marker.scale.z = 0.04
        pl_marker.color.a = 1
        pl_marker.color.r = color[0]
        pl_marker.color.g = color[1]
        pl_marker.color.b = color[2]
        pl_marker.lifetime = rospy.Duration.from_sec(1200)
        pl_marker.action = Marker.ADD
        pl_marker.points = self.array_to_pose_msg(point_list)

        self.marker_pub.publish(pl_marker)

    def append_point_cloud(self, path, name, point_list):
        p = pcl.PointCloud()
        p.from_file(path + name + ".pcd")
        a = np.asarray(p)
        print type(a), a.shape
        print type(point_list), point_list.shape

        new_a = np.concatenate((a,point_list), axis=0)
        new_p = pcl.PointCloud(new_a.astype(np.float32) )
        # p.from_array(new_a.astype(float))
        new_p.to_file(path + '/distribution/' + name + '_new.pcd')
        return p

    def get_xyz_from_point_cloud(self, path, name, max_idx_list):
        p = pcl.PointCloud()
        p.from_file(path + name + ".pcd")
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

    def load_img_mask(self, data_list, path):

        img_list = []
        mask_list = []

        for idx, data in enumerate(data_list):
            # print idx

            img_name = path + data.name + "_rgb.png"
            img = cv2_read_file_rgb(img_name)
            if img is None:
                print "[ERROR] No image"
                return None, None

            img = crop_to_center(img)
            img = cv2.resize(img, self.input_dims)
            img_list.append(img)

            mask_name = path + data.name + "_mask.png"
            mask = cv2.imread(mask_name)
            if mask is None:
                print "[ERROR] No mask"
                return None, None

            mask = crop_to_center(mask)
            mask = np.reshape(mask[:,:,0], (mask.shape[0], mask.shape[1]))
            mask_list.append(mask)

        return img_list, mask_list


    def load_conv5(self, img_list, mask_list):

        conv5_list = np.array([]).reshape([0] + list(self.net.blobs['conv5'].data.shape[1:]))

        for idx, img in enumerate(img_list):
            print idx,
            sys.stdout.flush()
            # print "img", img.shape
            self.net_proc_forward_layer(img, mask_list[idx])
            # self.net_preproc_forward(img)
            conv5_list = np.append(conv5_list, self.net.blobs['conv5'].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv5_list

    def load_layer_fix_filter(self, load_layer, fix_layer, fix_layer_data_list, img_list, mask_list, filter_idx):

        # layer_list = np.array([]).reshape([0] + list(self.net.blobs[load_layer].data.shape[1:]))
        # data_list = np.array([]).reshape([0] + list(self.net.blobs["data"].data.shape[1:]))

        layer_list = np.zeros([len(img_list)] + list(self.net.blobs[load_layer].data.shape[1:]))
        bp_list = np.zeros([len(img_list)] + list(self.net.blobs['data'].data.shape[2:]))

        for idx, img in enumerate(img_list):
            print idx,
            sys.stdout.flush()
            # print "img", img.shape
            self.net_proc_forward_layer(img, mask_list[idx])
            self.net_proc_backward_with_data(filter_idx, fix_layer_data_list[idx], fix_layer)
            layer_list[idx,:] = self.net.blobs[load_layer].diff
            bp_list[idx,:] = np.absolute(self.net.blobs['data'].diff[0,:]).mean(axis=0)
            # layer_list = np.append(layer_list, self.net.blobs[load_layer].diff, axis=0)
            # data_list[idx] = copy.deepcopy(self.net.blobs['data'].diff)
            # print "shape", self.net.blobs['conv5'].data.shape
        return layer_list, bp_list

    # def load_conv4_fix_conv5(self, img_list, mask_list, filter_idx):
    #
    #     conv4_list = np.array([]).reshape([0] + list(self.net.blobs['conv4'].data.shape[1:]))
    #
    #     for idx, img in enumerate(img_list):
    #         print idx, " ",
    #         # print "img", img.shape
    #         self.net_proc_forward_layer(img, mask_list[idx])
    #         self.net_proc_backward(filter_idx,'conv5')
    #
    #         conv4_list = np.append(conv4_list, self.net.blobs['conv4'].diff, axis=0)
    #         # print "shape", self.net.blobs['conv5'].data.shape
    #     return conv4_list

    # binaraizes such that output is a 1-d array where each entry is whether a filter fires, also ouputs the max value
    def binarize(self, data, threshold):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        max_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            max_value = np.amax(filter)
            if max_value > threshold:
                bin_data[id] = 1
            max_data[id] = max_value
        return bin_data, max_data

    def average(self, data):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            sum_value = np.sum(filter)
            bin_data[id] = sum_value / (data.shape[1]*data.shape[2])

        return bin_data

if __name__ == '__main__':
    rospy.init_node('data_monster', anonymous=True)

    ds = DataSettings()
    data_monster = DataMonster(settings, ds)
    path = settings.ros_dir + '/data/'
    data_monster.set_path(path)
    mode = 4

    # naming convention layer-palm or finger, xxx filter each layer, self or auto picked filters,
    # max or avg xy position, back prop single or all, seg_point_cloud or full, number_train, deconv or grad,
    # filter cm deviation, average width on point cloud, threshold, find max filter or most above threhold
    # name = '(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
    name = ds.get_name()
    # train
    if mode == 0:
        # name = '(4-p-3-f)_(1-2-[9-10])_auto_avg_sin_seg_42_g_bxy_10_(20-3-0.5)_of'
        dist_dic = data_monster.train_each_case()

        for case in dist_dic:
            dist_dic[case].save(settings.ros_dir + '/data/', "[" + case + "]" + name)
    # test
    elif mode == 1:
        data_monster.show_backprop = False
        distribution = Distribution()
        case1 = '[side_wrap:cylinder]'
        case2 = '[side_wrap:cuboid]'
        # name = '(4-p-3-f)_(1-2-[9-10])_auto_avg_all_seg_42_g_bxy_10_(20-10-2)_of_f3'
        distribution.load(settings.ros_dir + '/data/', case2 + name)

        data_list = [get_data_by_name(path,dl.data_name_list[16])]
        diff_avg_dic, diff_dist_dic, diff_fail = data_monster.test_accuracy(distribution, data_list)
        print diff_dist_dic

    # filter
    elif mode == 2:
        distribution = Distribution()
        case = '[side_wrap:cylinder]'
        distribution.load(settings.ros_dir + '/data/', case + name)
        new_dist = data_monster.filter_distribution(distribution, 0.03**2)
        new_dist.save(settings.ros_dir + '/data/', case + name + '_f3')
    # print names
    elif mode == 3:
        data_all = get_data_all(settings.ros_dir + '/data/')
        for i, data in enumerate(data_all):
            print "'" + data.name + "'" , ", # ", i

    elif mode == 4:

        data_monster.cross_validation(settings.ros_dir, name, False)

    elif mode == 5:
        dist1 = Distribution()
        case1 = '[side_wrap:cylinder]'
        dist1.load(settings.ros_dir + '/data/', case1 + name)

        dist2 = Distribution()
        case2 = '[side_wrap:cuboid]'
        dist2.load(settings.ros_dir + '/data/', case2 + name)

        dist1.merge(dist2)
        dist1.save(settings.ros_dir + '/data/', case1 + case2 + name)


    print "done"
    raw_input()
