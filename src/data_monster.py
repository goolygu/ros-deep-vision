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

import copy
import scipy
from scipy.special import expit

from data_util import *

from data_settings import *
from data_analyzer import *

from input_manager import *

from visualizer import *

class DataMonster:

    def __init__(self, settings, data_settings):
        print 'initialize'
        self.settings = settings
        self.ds = data_settings

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
        self.threshold['conv4'] = self.ds.thres_conv4
        self.threshold['conv3'] = self.ds.thres_conv3
        self.threshold['conv2'] = self.ds.thres_conv2
        self.threshold['conv1'] = 0
        self.visualizer = Visualizer()
        self.visualizer.set_frame("/r2/head/asus_depth_optical_frame")
        self.visualizer.set_topics(['grasp_distribution', 'feature', "grasp_target"])


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


        self.back_mode = self.ds.back_prop_mode
        w = self.ds.avg_pointcloud_width
        self.average_grid = np.mgrid[-w:w,-w:w]
        self.visualize = True
        self.show_backprop = True
        self.input_manager = InputManager(self.ds, self.input_dims)
        # self.input_manager.set_width(self.ds.input_width)
        # self.point_cloud_shape = (480,640)

    def set_box(self, min_max_box, margin_ratio):
        self.input_manager.set_box(min_max_box, margin_ratio)

    # def set_path(self, path):
    #     self.path = path

    def set_train_path(self, path):
        self.train_path = path

    def train_each_case(self, tbp):
        data_dic = self.input_manager.get_data_dic(self.train_path, dl.data_name_list)
        dist_dic = {}
        for case in data_dic:
            print "train case", case
            if tbp:
                dist_dic[case] = self.train(data_dic[case])
            else:
                dist_dic[case] = self.train_without_tbp(data_dic[case])
        return dist_dic

    def train_all(self):
        data_list = self.input_manager.get_data_all(self.train_path)
        return self.train(data_list)

    def train(self, data_list):
        distribution = Distribution()

        conv5_list = self.load_conv5(data_list)

        filter_idx_list_conv5 = self.find_consistent_filters(conv5_list, self.threshold["conv5"], self.ds.n_conv5_f)
        print "consistent filter conv5", filter_idx_list_conv5

        # distribution.set_tree_list([],filter_idx_list_conv5)
        # handle conv 4 filters
        for filter_idx_5 in filter_idx_list_conv5:

            print "handling filter", filter_idx_5, "layer conv5"
            # img_src is the abs diff back propagate to the image layer
            conv4_diff_list, img_src_5_list = self.load_layer_fix_filter_list("conv4", "conv5", conv5_list, data_list, filter_idx_5)
            rel_pos_5 = self.get_relative_pos(filter_idx_5, data_list, conv5_list, img_src_5_list, self.ds.frame_list_conv5, self.threshold["conv5"])
            self.set_distribution(distribution, self.ds.frame_list_conv5, filter_idx_5, rel_pos_5, [])

            filter_idx_list_conv4 = self.find_consistent_filters(conv4_diff_list, self.threshold["conv4"], self.ds.n_conv4_f)
            print "consistent filter conv4", filter_idx_list_conv4

            for i, filter_idx_4 in enumerate(filter_idx_list_conv4):
                print "handling filter", filter_idx_4, "layer conv4"
                conv3_diff_list, img_src_4_list = self.load_layer_fix_filter_list("conv3", "conv4", conv4_diff_list, data_list, filter_idx_4)
                rel_pos_4 = self.get_relative_pos(filter_idx_4, data_list, conv4_diff_list, img_src_4_list, self.ds.frame_list_conv4, self.threshold["conv4"])
                self.set_distribution(distribution, self.ds.frame_list_conv4, filter_idx_4, rel_pos_4, [filter_idx_5])

                filter_idx_list_conv3 = self.find_consistent_filters(conv3_diff_list, self.threshold["conv3"], self.ds.n_conv3_f)
                print "consistent filter conv3", filter_idx_list_conv3

                for j, filter_idx_3 in enumerate(filter_idx_list_conv3):
                    print "handling filter", filter_idx_3, "layer conv3"
                    conv2_diff_list, img_src_3_list = self.load_layer_fix_filter_list("conv2", "conv3", conv3_diff_list, data_list, filter_idx_3)
                    rel_pos_3 = self.get_relative_pos(filter_idx_3, data_list, conv3_diff_list, img_src_3_list, self.ds.frame_list_conv3, self.threshold["conv3"])
                    self.set_distribution(distribution, self.ds.frame_list_conv3, filter_idx_3, rel_pos_3, [filter_idx_5, filter_idx_4])

                    filter_idx_list_conv2 = self.find_consistent_filters(conv2_diff_list, self.threshold["conv2"], self.ds.n_conv2_f)
                    print "consistent filter conv2", filter_idx_list_conv2

                    for k, filter_idx_2 in enumerate(filter_idx_list_conv2):
                        print "handling filter", filter_idx_2, "layer conv2"
                        conv1_diff_list, img_src_2_list = self.load_layer_fix_filter_list("conv1", "conv2", conv2_diff_list, data_list, filter_idx_2)
                        rel_pos_2 = self.get_relative_pos(filter_idx_2, data_list, conv2_diff_list, img_src_2_list, self.ds.frame_list_conv2, self.threshold["conv2"])
                        self.set_distribution(distribution, self.ds.frame_list_conv2, filter_idx_2, rel_pos_2, [filter_idx_5, filter_idx_4, filter_idx_3])

                        # filter_idx_list_conv1 = self.find_consistent_filters(conv1_diff_list, self.threshold["conv1"], self.ds.n_conv1_f)
                        # print "consistent filter conv1", filter_idx_list_conv1


        return distribution

    def train_without_tbp(self, data_list):
        distribution = Distribution()

        conv5_list = self.load_layer(data_list, 'conv5')
        conv4_list = self.load_layer(data_list, 'conv4')
        conv3_list = self.load_layer(data_list, 'conv3')
        conv2_list = self.load_layer(data_list, 'conv2')

        filter_idx_list_conv5 = self.find_consistent_filters(conv5_list, self.threshold["conv5"], self.ds.n_conv5_f)
        print "consistent filter conv5", filter_idx_list_conv5

        filter_idx_list_conv4 = self.find_consistent_filters(conv4_list, self.threshold["conv4"], self.ds.n_conv4_f)
        print "consistent filter conv4", filter_idx_list_conv4

        filter_idx_list_conv3 = self.find_consistent_filters(conv3_list, self.threshold["conv3"], self.ds.n_conv3_f)
        print "consistent filter conv3", filter_idx_list_conv3

        filter_idx_list_conv2 = self.find_consistent_filters(conv2_list, self.threshold["conv2"], self.ds.n_conv2_f)
        print "consistent filter conv2", filter_idx_list_conv2

        # distribution.set_tree_list([],np.append(filter_idx_list_conv5, -1))
        # distribution.set_tree_list([-1],np.append(filter_idx_list_conv4, -1))
        # distribution.set_tree_list([-1, -1],np.append(filter_idx_list_conv3, -1))
        # distribution.set_tree_list([-1, -1, -1],filter_idx_list_conv2)
        distribution.set_tree_sig([-1,-1,-1])
        # img_src_fid_array_4 = np.zeros([len(data_list)] + [len(filter_idx_list_conv4)] + list(self.net.blobs["data"].diff.shape[2:]))
        # img_src_fid_array_3 = np.zeros([len(data_list)] + [len(filter_idx_list_conv3)] + list(self.net.blobs["data"].diff.shape[2:]))
        # img_src_fid_array_2 = np.zeros([len(data_list)] + [len(filter_idx_list_conv2)] + list(self.net.blobs["data"].diff.shape[2:]))

        for i, filter_idx_4 in enumerate(filter_idx_list_conv4):
            if filter_idx_4 == -1:
                continue
            print "handling filter", filter_idx_4, "layer conv4"
            conv3_diff_list, img_src_4_list = self.load_layer_fix_filter_list("conv3", "conv4", conv4_list, data_list, filter_idx_4)
            # img_src_fid_array_4[:,i,:] = img_src_4_list
            rel_pos_4 = self.get_relative_pos(filter_idx_4, data_list, conv4_list, img_src_4_list, self.ds.frame_list_conv4, self.threshold["conv4"])
            self.set_distribution(distribution, self.ds.frame_list_conv4, filter_idx_4, rel_pos_4, [-1])

        for j, filter_idx_3 in enumerate(filter_idx_list_conv3):
            print "handling filter", filter_idx_3, "layer conv3"
            conv2_diff_list, img_src_3_list = self.load_layer_fix_filter_list("conv2", "conv3", conv3_list, data_list, filter_idx_3)
            # img_src_fid_array_3[:,j,:] = img_src_3_list
            rel_pos_3 = self.get_relative_pos(filter_idx_3, data_list, conv3_list, img_src_3_list, self.ds.frame_list_conv3, self.threshold["conv3"])
            self.set_distribution(distribution, self.ds.frame_list_conv3, filter_idx_3, rel_pos_3, [-1, -1])

        for k, filter_idx_2 in enumerate(filter_idx_list_conv2):
            print "handling filter", filter_idx_2, "layer conv2"
            conv1_diff_list, img_src_2_list = self.load_layer_fix_filter_list("conv1", "conv2", conv2_list, data_list, filter_idx_2)
            # img_src_fid_array_3[:,j,:] = img_src_3_list
            rel_pos_2 = self.get_relative_pos(filter_idx_2, data_list, conv2_list, img_src_2_list, self.ds.frame_list_conv2, self.threshold["conv2"])
            self.set_distribution(distribution, self.ds.frame_list_conv2, filter_idx_2, rel_pos_2, [-1, -1, -1])

        # rel_pose_list_3 = self.get_relative_pos_list(filter_idx_list_conv3, data_list, conv3_list, img_src_fid_array_3, self.ds.frame_list_conv3, self.threshold["conv3"])
        # self.set_distribution(distribution, self.ds.frame_list_conv3, filter_idx_list_conv3, rel_pose_list_3, [-1, -1])
        #
        # rel_pose_list_4 = self.get_relative_pos_list(filter_idx_list_conv4, data_list, conv4_list, img_src_fid_array_4, self.ds.frame_list_conv4, self.threshold["conv4"])
        # self.set_distribution(distribution, self.ds.frame_list_conv4, filter_idx_list_conv4, rel_pose_list_4, [-1])

        return distribution

    def test_clutter(self, path, data_path, name, tbp, case_list, single_test):

        dist_path = path + '/distribution/'
        result_path = path + '/result/'
        result_xyz = {}
        result = {}
        fail_count = {}
        distribution_dic = {}

        # data_path = path + '/data/setclutter1/'
        if single_test == None:
            data_list = self.input_manager.get_data_all(data_path)#, [23])
        else:
            data_list = self.input_manager.get_data_all(data_path, [single_test])

        for case in case_list:
            distribution = Distribution()
            distribution.load(dist_path, "[" + case + "]" + name)
            distribution_dic[case] = self.filter_distribution(distribution, self.ds.filter_low_n)


        for i, data in enumerate(data_list):
            if self.visualize:
                cv2.destroyAllWindows()
                cv2.imshow("img", data.img)
                cv2.waitKey(100)
            case = data.action + ':' + data.target_type
            if not case in result:
                result[case] = {}
                result_xyz[case] = {}
                fail_count[case] = {}
            obj = data.name
            result_xyz[case][obj], result[case][obj], fail_count[case][obj] = self.test_accuracy(distribution_dic[case], [data], tbp)

            if self.visualize:
                print "continue?"
                key = raw_input()
                if key == 'n':
                    break

        test_name = self.ds.get_test_name()
        run_analysis(result)
        check_fail(fail_count)

        with open(result_path + "/" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

    def cross_validation(self, path, name, train, tbp):
        self.visualize = False

        data_name_dic = self.input_manager.get_data_name_dic(self.train_path, dl.data_name_list)
        dist_dic = {}
        result_xyz = {}
        result = {}
        fail_count = {}
        dist_path = path + '/distribution/'
        result_path = path + '/result/'

        if True:#self.ds.evaluate == 'full':
            full_distribution = {}

            for case in data_name_dic:
                if train == False and os.path.isfile(dist_path + "[" + case + "]" + name + ".yaml"):
                    full_distribution[case] = Distribution()
                    full_distribution[case].load(dist_path, "[" + case + "]" + name)
                else:
                    print "full training", case
                    train_data_list_full = []
                    for j, train_object in enumerate(data_name_dic[case]):
                        train_data_list_full = train_data_list_full + data_name_dic[case][train_object]

                    if tbp:
                        full_distribution[case] = self.train(train_data_list_full)
                    else:
                        full_distribution[case] = self.train_without_tbp(train_data_list_full)
                    full_distribution[case].save(dist_path, "[" + case + "]" + name)

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
            print "start training"
            for k, case in enumerate(data_name_dic):
                print "train case", case
                for i, test_object in enumerate(data_name_dic[case]):#enumerate(["katibox"]):#
                    print "leave", i, test_object, "out"
                    if tbp:
                        distribution = self.train(train_data_list[case][test_object])
                    else:
                        distribution = self.train_without_tbp(train_data_list[case][test_object])

                    # print "test_data_list", test_data_list
                    distribution.save(dist_path + '/cross_validation/', "[" + case + "][leave_" + test_object + "]" + name)


        print "start testing"
        # test
        for k, case in enumerate(data_name_dic):
            result[case] = {}
            result_xyz[case] = {}
            fail_count[case] = {}
            print "test case", case
            for i, test_object in enumerate(data_name_dic[case]):#enumerate(["katibox"]):#
                print "test", i, test_object
                distribution = Distribution()
                distribution.load(dist_path + '/cross_validation/', "[" + case + "][leave_" + test_object + "]" + name)

                if self.ds.evaluate == 'full':
                    for other_case in full_distribution:
                        if other_case != case:
                            distribution.merge(full_distribution[other_case])

                distribution = self.filter_distribution(distribution, self.ds.filter_low_n)
                result_xyz[case][test_object], result[case][test_object], fail_count[case][test_object] = self.test_accuracy(distribution, test_data_list[case][test_object], tbp)


        test_name = self.ds.get_test_name()
        run_analysis(result)
        check_fail(fail_count)

        with open(result_path + "/cross_validation_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

        with open(result_path + "/cross_validation_xyz_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(result_xyz, f, default_flow_style=False)

        with open(result_path + "/cross_validation_fail_" + name + "_" + test_name + '.yaml', 'w') as f:
            yaml.dump(fail_count, f, default_flow_style=False)


    def test_accuracy(self, distribution, data_list, tbp):

        diff_sum_dic = {}
        diff_count = {}
        diff_fail = {}

        for idx, data in enumerate(data_list):
            print data.name
            if tbp:
                filter_xyz_dict, filter_resp_dict = self.get_all_filter_xyz(data, distribution)
            else:
                filter_xyz_dict, filter_resp_dict = self.get_all_filter_xyz_notbp(data, distribution)

            distribution_cf = self.get_distribution_cameraframe(distribution, filter_xyz_dict)

            avg_dic = self.model_distribution(distribution_cf, filter_resp_dict)

            if self.visualize:
                self.show_feature(filter_xyz_dict)
                self.show_point_cloud(data.name)
                self.show_distribution(distribution_cf)
            # print "avg", avg_dic
            frame_gt_xyz = {}
            for frame in avg_dic:
                # get ground truth frame location
                frame_gt_xyz[frame] = np.array(self.get_frame_xyz(data, frame))
                if not frame in diff_sum_dic:
                    diff_sum_dic[frame] = np.array([0.,0.,0.])
                    diff_count[frame] = 0
                    diff_fail[frame] = 0
                # print "frame_xyz", frame_xyz
                if not np.isnan(avg_dic[frame][0]):
                    diff_sum_dic[frame] += np.absolute(frame_gt_xyz[frame] - avg_dic[frame])
                    diff_count[frame] += 1
                else:
                    diff_fail[frame] += 1
            if self.visualize:
                self.show_frames(frame_gt_xyz, "gt", None)

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

        data_list = [self.input_manager.get_data_by_name(self.train_path,dl.data_name_list[15])]
        # img_list, mask_list = self.load_img_mask_list(data_list, self.train_path)

        for idx, data in enumerate(data_list):
            print data.name

            filter_xyz_dict = self.get_all_filter_xyz(data, distribution)
            # self.show_feature(filter_xyz_dict)
            # print filter_xyz_dict
            distribution_cf = self.get_distribution_cameraframe(distribution, filter_xyz_dict)
            self.show_point_cloud(data.name)
            self.model_distribution(distribution_cf)
            self.show_distribution(distribution_cf)


    def filter_distribution(self, dist, low_n):

        if low_n < 0:
            return dist

        if self.ds.filter_same_parent:
            return self.filter_distribution_same_parent(dist, low_n)
        else:
            return self.filter_distribution_variance(dist, low_n)

    def filter_distribution_variance(self,dist, low_n):
        new_dist = Distribution()

        data_dict = dist.data_dict
        var_dic = Dic2()
        frame_dic = {}
        for sig in data_dict:
            for frame in data_dict[sig]:
                frame_dic[frame] = 1
                point_list = data_dict[sig][frame]
                # dist = np.linalg.norm(np.nanmean(point_list, axis = 0))
                var = np.linalg.norm(np.nanvar(point_list, axis = 0))
                # var_dist = dist * var
                var_dic.add(frame, sig, var)
                # var_dic.add(frame, sig, np.sum(var))

        print "low variance filters"
        for frame in frame_dic:
            var_list = var_dic.get_sublist(frame)
            # print var_list
            var_list = sorted(var_list, key=lambda var_tuple: var_tuple[1])
            print ""
            print frame
            if low_n < 1:
                low_n = int(math.floor(len(var_list) * low_n))
            for i in range(min(low_n, len(var_list))):
                sig = var_list[i][0]
                point_list = data_dict[sig][frame]
                new_dist.set(sig, frame, point_list)
                new_dist.set_tree_sig(sig)
                print sig,
        #         if np.sum(var) < threshold:
        #             new_dist.set(sig, frame, point_list)
        # new_dist.filter_tree = dist.filter_tree
        return new_dist

    def filter_distribution_same_parent(self, dist, low_n):

        data_dict = dist.data_dict
        var_dic = {}
        frame_dic = {}
        for sig in data_dict:
            parent = sig[0]
            if not parent in var_dic:
                var_dic[parent] = Dic2()
            for frame in data_dict[sig]:
                frame_dic[frame] = 1
                point_list = data_dict[sig][frame]
                # dist = np.linalg.norm(np.nanmean(point_list, axis = 0))
                var = np.linalg.norm(np.nanvar(point_list, axis = 0))
                # var_dist = dist * var
                var_dic[parent].add(frame, sig, var)

        dist_dic = {}
        min_average = float('inf')
        min_parent = -1
        parent_var_list = []
        print "low variance filters"
        for parent in var_dic:
            print "parent", parent
            dist_dic[parent] = Distribution()
            var_sum = 0
            count = 0
            for frame in frame_dic:
                var_list = var_dic[parent].get_sublist(frame)
                # print var_list
                var_list = sorted(var_list, key=lambda var_tuple: var_tuple[1])
                print ""
                print frame
                for i in range(min(low_n, len(var_list))):
                    var_sum += var_list[i][1]
                    count += 1
                    sig = var_list[i][0]
                    point_list = data_dict[sig][frame]
                    dist_dic[parent].set(sig, frame, point_list)
                    dist_dic[parent].set_tree_sig(sig)
                    print sig,
            var_average = var_sum / count
            if var_average < min_average:
                min_average = var_average
                min_parent = parent

            parent_var_list.append((parent, var_average))

        # parent_var_list = sorted(parent_var_list, key=lambda var_tuple: var_tuple[1])
        # merge_dist = Distribution()
        # for i in range(1):
        #     merge_dist.merge(dist_dic[parent_var_list[i][0]])
        # return merge_dist

        return dist_dic[min_parent]

    def show_frames(self, frames_xyz, name, color_map):
        if color_map == None:
            color_map = {}
            color_map["r2/left_palm"] = (0.5,0.5,0)
            color_map["r2/left_thumb_tip"] = (0.5,0.5,0)
            color_map["r2/left_index_tip"] = (0.5,0.5,0)

        if self.visualize:
            for frame in frames_xyz:
                ns = name + frame
                self.visualizer.publish_point_array([frames_xyz[frame]], 0, ns, "grasp_target", color_map[frame], Marker.SPHERE_LIST, 1, 0.04 )

    def model_distribution(self, dist_cf, resp_dict):
        dist_list = {}
        dist_list["r2/left_palm"] = np.array([]).reshape([0,3])
        dist_list["r2/left_thumb_tip"] = np.array([]).reshape([0,3])
        dist_list["r2/left_index_tip"] = np.array([]).reshape([0,3])

        w_list = {}
        w_list["r2/left_palm"] = np.array([])
        w_list["r2/left_thumb_tip"] = np.array([])
        w_list["r2/left_index_tip"] = np.array([])
        print resp_dict

        # concatenate all points in camera frame of same robot joint
        for sig in dist_cf:
            for frame in dist_cf[sig]:
                # print type(dist_cf[sig][frame])
                # remove nan
                dist = dist_cf[sig][frame]
                nan_mask = np.any(np.isnan(dist), axis=1)
                dist = dist[~nan_mask]
                if self.ds.dist_to_grasp_point == "weightmean" or self.ds.dist_to_grasp_point == "weightdensepoint":
                    weight = np.ones(dist.shape[0]) * resp_dict[sig]
                else:
                    weight = np.ones(dist.shape[0]) * (resp_dict[sig]/dist.shape[0])
                dist_list[frame] = np.concatenate((dist_list[frame], dist), axis=0)
                w_list[frame] = np.concatenate((w_list[frame], weight), axis = 0)

        # print dist_list

        avg_dic = {}
        for frame in dist_list:
            if self.ds.dist_to_grasp_point == "mean":
                avg_dic[frame] = np.nanmean(dist_list[frame], axis=0)
            elif self.ds.dist_to_grasp_point == "density":
                avg_dic[frame] = find_max_density(dist_list[frame])
            elif self.ds.dist_to_grasp_point == "densepoint":
                avg_dic[frame] = find_max_density_point(dist_list[frame])
            elif self.ds.dist_to_grasp_point == "weightmean" or self.ds.dist_to_grasp_point == "filterweightmean":
                if sum(w_list[frame]) == 0:
                    print "weights sum to zero", dist_list[frame]
                    avg_dic[frame] = np.nanmean(dist_list[frame], axis=0)
                else:
                    avg_dic[frame] = np.average(dist_list[frame], axis=0, weights=w_list[frame])
            elif self.ds.dist_to_grasp_point == "weightdensepoint":
                if sum(w_list[frame]) == 0:
                    print "weights sum to zero"
                    avg_dic[frame] = np.nanmean(dist_list[frame], axis=0)
                else:
                    avg_dic[frame] = find_weighted_max_density_point(dist_list[frame], w_list[frame])


        color_map = {}
        color_map["r2/left_palm"] = (0.5,0,0)
        color_map["r2/left_thumb_tip"] = (0,0.5,0)
        color_map["r2/left_index_tip"] = (0,0,0.5)

        self.show_frames(avg_dic, "", color_map)

        return avg_dic

    def set_distribution(self, distribution, frame_list, filter_idx, rel_pos_list, parent_filters):
        for j, frame in enumerate(frame_list):
            distribution.set(parent_filters + [filter_idx], frame, rel_pos_list[j,:,:])
        distribution.set_tree(parent_filters, filter_idx)
    # def set_distribution(self, distribution, frame_list, filter_idx_list, dist_list, parent_filters):
    #     for j, frame in enumerate(frame_list):
    #         for i, filter_idx in enumerate(filter_idx_list):
    #             distribution.set(parent_filters + [filter_idx], frame, dist_list[j,i,:,:])

    def show_distribution(self, dist_cf):
        color_map = {}
        color_map["r2/left_palm"] = (1,0,0)
        color_map["r2/left_thumb_tip"] = (0,1,0)
        color_map["r2/left_index_tip"] = (0,0,1)

        idx = 0
        for sig in dist_cf:
            for frame in dist_cf[sig]:
                ns = "/" + "/".join([str(c) for c in sig]) + "-" + frame
                self.visualizer.publish_point_array(dist_cf[sig][frame], idx, ns, 'grasp_distribution', color_map[frame], Marker.POINTS, 0.4, 0.01 )


    def show_feature(self, filter_xyz_dict):
        color_map = {}
        color_map[1] = (1,1,0)
        color_map[2] = (0,1,1)
        color_map[3] = (1,0,1)
        color_map[4] = (1,0.5,0.5)

        for sig in filter_xyz_dict:
            print sig, filter_xyz_dict[sig]

        idx = 0
        for sig in filter_xyz_dict:
            ns = "/" + "/".join([str(c) for c in sig])
            self.visualizer.publish_point_array([filter_xyz_dict[sig]], idx, ns, 'feature', color_map[len(sig)], Marker.POINTS, 0.9, 0.01 )

    # add offset from feature xyz to grasp_points
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

    def get_filter_xyz(self, layer_data, pc, threshold):
        resize_ratio = float(pc.shape[0]) / float(layer_data.shape[0])
        if True:
            filter_xy = self.get_filter_avg_xy(layer_data, threshold)
        else:
            filter_xy = self.get_filter_max_xy(layer_data, threshold)
        orig_xy = self.get_orig_xy(filter_xy, resize_ratio)
        filter_xyz = self.get_average_xyz_from_point_cloud(pc, [orig_xy], self.average_grid)
        return filter_xyz[0], filter_xy

    # def get_filter_xyz(self, layer_data, pc_array, threshold):
    #     image_size_orig = self.input_manager.get_after_crop_size()
    #     resize_ratio = float(image_size_orig[0]) / float(layer_data.shape[0])
    #     if True:
    #         filter_xy = self.get_filter_avg_xy(layer_data, threshold)
    #     else:
    #         filter_xy = self.get_filter_max_xy(layer_data, threshold)
    #     orig_xy = self.get_orig_xy(filter_xy, resize_ratio)
    #     filter_xyz = self.get_average_xyz_from_point_cloud_array(pc_array, [orig_xy], self.average_grid)
    #     return filter_xyz[0], filter_xy


    def get_all_filter_xyz_notbp(self, data, dist):

        xyz_dict = {}
        response_dict = {}
        self.net_proc_forward_layer(data.img, data.mask)
        conv5_data = copy.deepcopy(self.net.blobs['conv5'].data[0])
        conv4_data = copy.deepcopy(self.net.blobs['conv4'].data[0])
        conv3_data = copy.deepcopy(self.net.blobs['conv3'].data[0])
        conv2_data = copy.deepcopy(self.net.blobs['conv2'].data[0])

        filter_idx_5_list = []
        if self.ds.filter_test == 'top':
            filter_idx_5_list = self.get_top_filters_in_list(conv5_data, dist.filter_tree, self.ds.conv5_top)
        elif self.ds.filter_test == 'all':
            filter_idx_5_list = dist.filter_tree

        for filter_idx_5 in filter_idx_5_list:
            if filter_idx_5 == -1:
                continue

            print 'conv5', filter_idx_5
            layer = 'conv5'

            if not self.filter_response_pass_threshold(conv5_data[filter_idx_5], self.ds.thres_conv5_test):
                continue

            conv4_data, img_src_5 = self.load_layer_fix_filter('conv4', 'conv5', conv5_data, data, filter_idx_5)
            xyz_dict[(filter_idx_5,)], max_xy = self.get_filter_xyz(img_src_5, data.pc, 0)
            response_dict[(filter_idx_5,)] = self.get_max_filter_response(conv5_data[filter_idx_5])

            self.show_gradient(str((filter_idx_5)), self.net.blobs['data'], max_xy, 0)


        filter_idx_4_list = []
        if not -1 in dist.filter_tree:
            filter_idx_4_list = []
        if self.ds.filter_test == 'top':
            filter_idx_4_list = self.get_top_filters_in_list(conv4_data, dist.filter_tree[-1], self.ds.conv4_top)
        elif self.ds.filter_test == 'all':
            filter_idx_4_list = dist.filter_tree[-1]

        for filter_idx_4 in filter_idx_4_list:
            if filter_idx_4 == -1:
                continue
            print 'conv4', filter_idx_4
            layer = 'conv4'

            if not self.filter_response_pass_threshold(conv4_data[filter_idx_4], self.ds.thres_conv4_test):
                continue

            conv3_data, img_src_4 = self.load_layer_fix_filter('conv3', 'conv4', conv4_data, data, filter_idx_4)
            xyz_dict[(-1, filter_idx_4)], max_xy = self.get_filter_xyz(img_src_4, data.pc, 0)
            response_dict[(-1, filter_idx_4)] = self.get_max_filter_response(conv4_data[filter_idx_4])

            self.show_gradient(str((-1, filter_idx_4)), self.net.blobs['data'], max_xy, 0)


        filter_idx_3_list = []
        if not -1 in dist.filter_tree[-1]:
            filter_idx_3_list = []
        elif self.ds.filter_test == 'top':
            filter_idx_3_list = self.get_top_filters_in_list(conv3_data, dist.filter_tree[-1][-1], self.ds.conv3_top)
        elif self.ds.filter_test == 'all':
            filter_idx_3_list = dist.filter_tree[-1][-1]

        for filter_idx_3 in filter_idx_3_list:
            print 'conv3', filter_idx_3
            layer = 'conv3'

            if not self.filter_response_pass_threshold(conv3_data[filter_idx_3], self.ds.thres_conv3_test):
                continue

            conv2_data, img_src_3 = self.load_layer_fix_filter('conv2', 'conv3', conv3_data, data, filter_idx_3)
            xyz_dict[(-1, -1, filter_idx_3)], max_xy = self.get_filter_xyz(img_src_3, data.pc, 0)
            response_dict[(-1, -1, filter_idx_3)] = self.get_max_filter_response(conv3_data[filter_idx_3])

            self.show_gradient(str((-1, -1, filter_idx_3)), self.net.blobs['data'], max_xy, 0)

        print "dist", dist.filter_tree

        filter_idx_2_list = []
        if not -1 in dist.filter_tree[-1][-1]:
            filter_idx_2_list = []
        elif self.ds.filter_test == 'top':
            filter_idx_2_list = self.get_top_filters_in_list(conv2_data, dist.filter_tree[-1][-1][-1], self.ds.conv2_top)
        elif self.ds.filter_test == 'all':
            filter_idx_2_list = dist.filter_tree[-1][-1][-1]

        for filter_idx_2 in filter_idx_2_list:
            print 'conv2', filter_idx_2
            layer = 'conv2'

            if not self.filter_response_pass_threshold(conv3_data[filter_idx_2], self.ds.thres_conv2_test):
                continue

            conv1_data, img_src_2 = self.load_layer_fix_filter('conv1', 'conv2', conv2_data, data, filter_idx_2)
            xyz_dict[(-1, -1, -1, filter_idx_2)], max_xy = self.get_filter_xyz(img_src_2, data.pc, 0)
            response_dict[(-1, -1, -1, filter_idx_2)] = self.get_max_filter_response(conv2_data[filter_idx_2])

            self.show_gradient(str((-1, -1, -1, filter_idx_2)), self.net.blobs['data'], max_xy, 0)

        return xyz_dict, response_dict

    def get_all_filter_xyz(self, data, dist):

        xyz_dict = {}
        response_dict = {}
        self.net_proc_forward_layer(data.img, data.mask)
        conv5_data = copy.deepcopy(self.net.blobs['conv5'].data[0,:])
        conv4_data = copy.deepcopy(self.net.blobs['conv4'].data[0])
        conv3_data = copy.deepcopy(self.net.blobs['conv3'].data[0])
        conv2_data = copy.deepcopy(self.net.blobs['conv2'].data[0])

        filter_idx_5_list = []
        if self.ds.filter_test == 'top':
            filter_idx_5_list = self.get_top_filters_in_list(conv5_data, dist.filter_tree, self.ds.conv5_top)
        elif self.ds.filter_test == 'all':
            filter_idx_5_list = dist.filter_tree

        for filter_idx_5 in filter_idx_5_list:
            print filter_idx_5

            if not self.filter_response_pass_threshold(conv5_data[filter_idx_5], self.ds.thres_conv5_test):
                continue

            conv4_diff, img_src_5 = self.load_layer_fix_filter('conv4', 'conv5', conv5_data, data, filter_idx_5)
            if not self.ds.tbp_test:
                conv4_diff = conv4_data
            xyz_dict[(filter_idx_5,)], max_xy = self.get_filter_xyz(img_src_5, data.pc, 0)
            response_dict[(filter_idx_5,)] = self.get_max_filter_response(conv5_data[filter_idx_5])

            self.show_gradient(str((filter_idx_5)), self.net.blobs['data'], max_xy, 0)
            # self.show_depth(str((filter_idx_5))+'depth', self.net.blobs['data'].data, pc_array)

            filter_idx_4_list = []
            if self.ds.filter_test == 'top':
                filter_idx_4_list = self.get_top_filters_in_list(conv4_diff, dist.filter_tree[filter_idx_5], self.ds.conv4_top)
            elif self.ds.filter_test == 'all':
                filter_idx_4_list = dist.filter_tree[filter_idx_5]

            for filter_idx_4 in filter_idx_4_list:
                print filter_idx_5, filter_idx_4

                if not self.filter_response_pass_threshold(conv4_diff[filter_idx_4], self.ds.thres_conv4_test):
                    continue

                conv3_diff, img_src_4 = self.load_layer_fix_filter('conv3', 'conv4', conv4_diff, data, filter_idx_4)
                if not self.ds.tbp_test:
                    conv3_diff = conv3_data
                xyz_dict[(filter_idx_5, filter_idx_4)], max_xy = self.get_filter_xyz(img_src_4, data.pc, 0)
                response_dict[(filter_idx_5, filter_idx_4)] = self.get_max_filter_response(conv4_diff[filter_idx_4])

                self.show_gradient(str((filter_idx_5, filter_idx_4)), self.net.blobs['data'], max_xy, 0)


                filter_idx_3_list = []
                if self.ds.filter_test == 'top':
                    filter_idx_3_list = self.get_top_filters_in_list(conv3_diff, dist.filter_tree[filter_idx_5][filter_idx_4], self.ds.conv3_top)
                elif self.ds.filter_test == 'all':
                    filter_idx_3_list = dist.filter_tree[filter_idx_5][filter_idx_4]

                for filter_idx_3 in filter_idx_3_list:
                    print filter_idx_5, filter_idx_4, filter_idx_3

                    if not self.filter_response_pass_threshold(conv3_diff[filter_idx_3], self.ds.thres_conv3_test):
                        continue

                    conv2_diff, img_src_3 = self.load_layer_fix_filter('conv2', 'conv3', conv3_diff, data, filter_idx_3)
                    if not self.ds.tbp_test:
                        conv2_diff = conv2_data

                    xyz_dict[(filter_idx_5, filter_idx_4, filter_idx_3)], max_xy = self.get_filter_xyz(img_src_3, data.pc, 0)
                    response_dict[(filter_idx_5, filter_idx_4, filter_idx_3)] = self.get_max_filter_response(conv3_diff[filter_idx_3])

                    self.show_gradient(str((filter_idx_5, filter_idx_4, filter_idx_3)), self.net.blobs['data'], max_xy, 0)

                    filter_idx_2_list = []
                    if self.ds.filter_test == 'top':
                        filter_idx_2_list = self.get_top_filters_in_list(conv2_diff, dist.filter_tree[filter_idx_5][filter_idx_4][filter_idx_3], self.ds.conv2_top)
                    elif self.ds.filter_test == 'all':
                        filter_idx_2_list = dist.filter_tree[filter_idx_5][filter_idx_4][filter_idx_3]

                    for filter_idx_2 in filter_idx_2_list:
                        print filter_idx_5, filter_idx_4, filter_idx_3, filter_idx_2

                        if not self.filter_response_pass_threshold(conv2_diff[filter_idx_2], self.ds.thres_conv2_test):
                            continue
                        conv1_data, img_src_2 = self.load_layer_fix_filter('conv1', 'conv2', conv2_diff, data, filter_idx_2)
                        xyz_dict[(filter_idx_5, filter_idx_4, filter_idx_3, filter_idx_2)], max_xy = self.get_filter_xyz(img_src_2, data.pc, 0)
                        response_dict[(filter_idx_5, filter_idx_4, filter_idx_3, filter_idx_2)] = self.get_max_filter_response(conv2_diff[filter_idx_2])

                        self.show_gradient(str((filter_idx_5, filter_idx_4, filter_idx_3, filter_idx_2)), self.net.blobs['data'], max_xy, 0)


        return xyz_dict, response_dict

    # dist is expected features
    def get_state(self, dist, data):

        xyz_dict = {}
        response_dict = {}
        self.net_proc_forward_layer(data.img, data.mask)
        conv5_data = copy.deepcopy(self.net.blobs['conv5'].data[0])

        # self.show_depth('depth', self.net.blobs['data'].data, pc_array)

        if not dist == None:
            filter_idx_5_list = dist.filter_tree
        else:
            if self.ds.filters == 'top':
                filter_idx_5_list = self.get_top_filters(conv5_data, self.ds.conv5_top)
            elif self.ds.filters == 'spread':
                filter_idx_5_list = self.get_spread_filters(conv5_data, self.ds.conv5_top)


        for filter_idx_5 in filter_idx_5_list:
            print filter_idx_5

            conv4_data, img_src_5 = self.load_layer_fix_filter('conv4', 'conv5', conv5_data, data, filter_idx_5)
            xyz_dict[(filter_idx_5,)], max_xy = self.get_filter_xyz(img_src_5, data.pc, 0)
            response_dict[(filter_idx_5,)] = self.get_max_filter_response(conv5_data[filter_idx_5])

            self.show_gradient(str((filter_idx_5)), self.net.blobs['data'], max_xy, 0)

            if not dist == None:
                filter_idx_4_list = dist.filter_tree[filter_idx_5]
            else:
                if self.ds.filters == 'top':
                    filter_idx_4_list =  self.get_top_filters(conv4_data, self.ds.conv4_top)
                elif self.ds.filters == 'spread':
                    filter_idx_4_list = self.get_spread_filters(conv4_data, self.ds.conv4_top)

            for filter_idx_4 in filter_idx_4_list:
                print filter_idx_5, filter_idx_4

                conv3_data, img_src_4 = self.load_layer_fix_filter('conv3', 'conv4', conv4_data, data, filter_idx_4)
                xyz_dict[(filter_idx_5, filter_idx_4)], max_xy = self.get_filter_xyz(img_src_4, data.pc, 0)
                response_dict[(filter_idx_5, filter_idx_4)] = self.get_max_filter_response(conv4_data[filter_idx_4])

                self.show_gradient(str((filter_idx_5, filter_idx_4)), self.net.blobs['data'], max_xy, 0)

                if not dist == None:
                    filter_idx_3_list = dist.filter_tree[filter_idx_5][filter_idx_4]
                else:
                    if self.ds.filters == 'top':
                        filter_idx_3_list = self.get_top_filters(conv3_data[0], self.ds.conv3_top)
                    elif self.ds.filters == 'spread':
                        filter_idx_3_list = self.get_spread_filters(conv3_data[0], self.ds.conv3_top)

                for filter_idx_3 in filter_idx_3_list:
                    print filter_idx_5, filter_idx_4, filter_idx_3

                    conv2_data, img_src_3 = self.load_layer_fix_filter('conv2', 'conv3', conv3_data, data, filter_idx_3)
                    xyz_dict[(filter_idx_5, filter_idx_4, filter_idx_3)], max_xy = self.get_filter_xyz(img_src_3, data.pc, 0)
                    response_dict[(filter_idx_5, filter_idx_4, filter_idx_3)] = self.get_max_filter_response(conv3_data[filter_idx_3])

                    self.show_gradient(str((filter_idx_5, filter_idx_4, filter_idx_3)), self.net.blobs['data'], max_xy, 0)

        return xyz_dict, response_dict


    def show_depth(self, name, layer_data, pc):

        img = layer_data[0]
        img = img.transpose((1,2,0))
        img = norm01c(img, 0)
        img_size = layer_data[0].shape[1]
        # image_size_orig = self.input_manager.get_after_crop_size()
        resize_ratio = float(pc.shape[0]) / float(img_size)

        for x in range(0, img_size):
            for y in range(0, img_size):

                orig_xy = self.get_orig_xy([x,y], resize_ratio)
                # m_idx = np.array([[round(orig_xy[0])],[round(orig_xy[1])]])
                # print "m idx", m_idx
                # xy_index = np.ravel_multi_index(m_idx.astype(int),(480,640))
                # print "xy", xy_index
                if np.isnan(pc[orig_xy[0],orig_xy[1]]):
                    img[x,y] = [1,0,0]
                # else:
                #     img[x,y] = [0,0,pc_array[xy_index[0]][2]]
        img = norm01c(img, 0)
        cv2.imshow(name, img)
        cv2.waitKey(100)

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


    def get_filter_max_xy(self, filter_response, threshold):
        assert filter_response.ndim == 2, "filter size incorrect"
        max_value = np.amax(filter_response)
        if max_value <= threshold:
            return np.array([float('nan'),float('nan')])
        max_idx = np.argmax(filter_response, axis=None)
        max_xy = np.unravel_index(max_idx, filter_response.shape)
        return max_xy

    def get_max_filter_response(self, filter_response):
        return np.nanmax(filter_response)

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

    def get_top_filters(self, layer_response, number):
        num_filter = layer_response.shape[0]
        max_list = np.zeros(num_filter)
        for filter_id in range(num_filter):
            max_list[filter_id] = np.amax(layer_response[filter_id])

        sorted_filter_idx_list = np.argsort(max_list)[::-1]
        sorted_filter_idx_list = sorted_filter_idx_list[0:number]

        return sorted_filter_idx_list

    def get_spread_filters(self, layer_response, number):

        response = copy.deepcopy(layer_response)
        max_list = np.zeros(number).astype(int)
        layer_height = response.shape[1]
        layer_width = response.shape[2]

        for i in range(number):
            max_flat_index = np.argmax(response)
            # print "max_flat_index", max_flat_index
            max_index = np.unravel_index(max_flat_index, response.shape)

            filter_id = int(max_index[0])
            filter_x = max_index[1]
            filter_y = max_index[2]
            max_list[i] = filter_id

            # inhibit close filters
            response[filter_id] *= 0
            for x in range(-5,6):
                for y in range(-5,6):
                    idx_x = x+filter_x
                    idx_y = y+filter_y
                    if idx_x < 0 or idx_x >= layer_height or idx_y < 0 or idx_y >= layer_width:
                        continue
                    dist_square = float(x**2 + y**2)
                    response[:,x+filter_x,y+filter_y] *= dist_square/(50.0+dist_square)

        return max_list

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

    def net_proc_deconv_with_data(self, filter_idx, data, backprop_layer):

        diffs = self.net.blobs[backprop_layer].diff * 0
        if self.ds.backprop_xy == 'sin':
            x,y = np.unravel_index(np.argmax(data[filter_idx]), data[filter_idx].shape)
            diffs[0][filter_idx][x][y] = data[filter_idx][x][y]
        elif self.ds.backprop_xy == 'all':
            diffs[0][filter_idx] = data[filter_idx]
        assert self.back_mode in ('grad', 'deconv')
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


    def get_average_xyz_from_point_cloud(self, pc, max_xy_list, receptive_grid):
        pc_array = pc.reshape((pc.shape[0]*pc.shape[1], pc.shape[2]))
        output = []
        for xy in max_xy_list:
            if np.isnan(xy[0]):
                output.append([float('nan'),float('nan'),float('nan')])
                print "filter response zero no max xy", xy
                continue
            # receptive grid has shape (2,w,w) that contains the grid x idx and y idx
            grid = np.zeros(receptive_grid.shape)
            grid[0] = xy[0] + receptive_grid[0]
            grid[1] = xy[1] + receptive_grid[1]

            # this step flattens to 2 arrays of x coordinates and y coordinates
            xy_receptive_list =np.reshape(grid, [2,-1])
            idx_receptive_list = np.ravel_multi_index(xy_receptive_list.astype(int),pc.shape[0:2])
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


    def load_conv5(self, data_list):#img_list, mask_list):

        conv5_list = np.array([]).reshape([0] + list(self.net.blobs['conv5'].data.shape[1:]))

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()
            # print "img", img.shape
            self.net_proc_forward_layer(data.img, data.mask)
            # self.net_preproc_forward(img)
            conv5_list = np.append(conv5_list, self.net.blobs['conv5'].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv5_list

    def load_layer(self, data_list, layer):#img_list, mask_list):

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

    def load_conv4_conv3(self, data_list):

        conv4_list = np.array([]).reshape([0] + list(self.net.blobs['conv4'].data.shape[1:]))
        conv3_list = np.array([]).reshape([0] + list(self.net.blobs['conv3'].data.shape[1:]))

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()
            # print "img", img.shape
            self.net_proc_forward_layer(data.img, data.mask)
            # self.net_preproc_forward(img)
            conv4_list = np.append(conv4_list, self.net.blobs['conv4'].data, axis=0)
            conv3_list = np.append(conv3_list, self.net.blobs['conv3'].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv4_list, conv3_list


    def load_layer_fix_filter_list(self, load_layer, fix_layer, fix_layer_data_list, data_list, filter_idx):

        layer_diff_list = np.zeros([len(data_list)] + list(self.net.blobs[load_layer].diff.shape[1:]))
        img_src_list = np.zeros([len(data_list)] + list(self.net.blobs['data'].data.shape[2:]))

        for idx, data in enumerate(data_list):
            print idx,
            sys.stdout.flush()

            layer_diff_list[idx,:], img_src_list[idx,:] = self.load_layer_fix_filter(load_layer, fix_layer, fix_layer_data_list[idx], data, filter_idx)

        return layer_diff_list, img_src_list

    # perform forward path and backward path while zeroing out all filter response except for filter_idx
    # return layer_diff_list which is the load layer diff and img_src_list which is the abs diff if back propagate to image layer
    def load_layer_fix_filter(self, load_layer, fix_layer, fix_layer_diff, data, filter_idx):
        self.net_proc_forward_layer(data.img, data.mask)
        self.net_proc_backward_with_data(filter_idx, fix_layer_diff, fix_layer)
        layer_diff = self.net.blobs[load_layer].diff[0,:]
        # mean is to average over all filters
        if self.ds.img_src_loc == "absolute":
            img_src = np.absolute(self.net.blobs['data'].diff[0,:]).mean(axis=0)
        elif self.ds.img_src_loc == "relu":
            img_src = np.maximum(self.net.blobs['data'].diff[0,:],0).mean(axis=0)
        # make a copy
        layer_diff = copy.deepcopy(layer_diff)
        img_src = copy.deepcopy(img_src)
        return layer_diff, img_src

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

    def average(self, data):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            sum_value = np.sum(filter)
            bin_data[id] = sum_value / (data.shape[1]*data.shape[2])

        return bin_data

if __name__ == '__main__':
    rospy.init_node('data_monster', anonymous=True)

    # tbp = False#True#
    ds = DataSettings()
    # ds.tbp = tbp
    tbp = ds.tbp
    data_monster = DataMonster(settings, ds)
    # path = settings.ros_dir + '/data/'
    if ds.dataset == "set1":
        import data_list_set1 as dl
    elif ds.dataset == "set2":
        import data_list_set2 as dl
    elif ds.dataset == "set3":
        import data_list_set3 as dl
    elif ds.dataset == "set4":
        import data_list_set4 as dl
    elif ds.dataset == "set5":
        import data_list_set5 as dl
    elif ds.dataset == "set6":
        import data_list_set6 as dl
    elif ds.dataset == "set7":
        import data_list_set7 as dl

    train_path = settings.ros_dir + '/data/' + ds.dataset + "/"
    # data_monster.set_path(path)
    data_monster.set_train_path(train_path)
    mode = 1

    dist_path = settings.ros_dir + '/distribution/'

    # naming convention layer-palm or finger, xxx filter each layer, self or auto picked filters,
    # max or avg xy position, back prop single or all, seg_point_cloud or full, number_train, deconv or grad,
    # filter cm deviation, average width on point cloud, threshold, find max filter or most above threhold
    # name = '(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
    name = ds.get_name()
    # train
    if mode == 0:
        # name = '(4-p-3-f)_(1-2-[9-10])_auto_avg_sin_seg_42_g_bxy_10_(20-3-0.5)_of'
        dist_dic = data_monster.train_each_case(tbp)

        for case in dist_dic:
            dist_dic[case].save(dist_path, "[" + case + "]" + name)
    # test
    elif mode == 1:
        data_monster.show_backprop = True#False#
        distribution = Distribution()
        case1 = '[side_wrap:cylinder]'
        case2 = '[side_wrap:cuboid]'
        # name = '(4-p-3-f)_(1-2-[9-10])_auto_avg_all_seg_42_g_bxy_10_(20-10-2)_of_f3'
        distribution.load(dist_path + '/cross_validation/', case1 + '[leave_yellowjar]' + name)

        # distribution.load(dist_path, case2 + name)
        distribution = data_monster.filter_distribution(distribution, ds.filter_low_n)

        data_monster.input_manager.set_visualize(True)
        data_list = [data_monster.input_manager.get_data_by_name(train_path,dl.data_name_list[112])]
        diff_avg_dic, diff_dist_dic, diff_fail = data_monster.test_accuracy(distribution, data_list, tbp)
        print diff_dist_dic

    # filter
    elif mode == 2:
        distribution = Distribution()
        case = '[side_wrap:cylinder]'
        distribution.load(dist_path, case + name)
        new_dist = data_monster.filter_distribution(distribution, 3)
        new_dist.save(dist_path, case + name + '_f3')
    # print names
    elif mode == 3:
        name_list = data_monster.input_manager.get_data_name_all(train_path)
        for i, name in enumerate(name_list):
            print "'" + name + "'" , ", # ", i
    # cross validation
    elif mode == 4:

        retrain = False#True#
        data_monster.cross_validation(settings.ros_dir, name, retrain, tbp)

    # merge
    elif mode == 5:
        dist1 = Distribution()
        case1 = '[side_wrap:cylinder]'
        dist1.load(dist_path, case1 + name)

        dist2 = Distribution()
        case2 = '[side_wrap:cuboid]'
        dist2.load(dist_path, case2 + name)

        dist1.merge(dist2)
        dist1.save(dist_path, case1 + case2 + name)

    elif mode == 7:
        data_monster.visualize = False#True
        data_monster.show_backprop = False
        case1 = 'side_wrap:cylinder'
        case2 = 'side_wrap:cuboid'

        data_monster.test_clutter(settings.ros_dir, settings.ros_dir + '/data/setclutter/', name, tbp, [case1, case2], None)
    print "done"
    raw_input()
