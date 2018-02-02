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

import yaml
from data_collector import Data
from data_ploter import *
from distribution import *
import pcl

from ros_deep_vision.srv import String2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import copy
import scipy

from data_util import *

from data_analyzer import *

from input_manager import *

from visualizer import *
from backprop_info import *
from hier_cnn_net import *

class DataSheep:

    def __init__(self, settings, data_settings):
        print 'initialize'
        self.settings = settings

        self.visualizer = Visualizer()
        self.visualizer.set_frame("/r2/head/asus_depth_optical_frame")
        self.visualizer.set_topics(['grasp_distribution', 'feature', "grasp_target"])

        self.net = HierCNNNet(settings, data_settings)

        self.set_data_settings(data_settings)
        self.set_visualize(True)
        self.net.show_backprop = True

    def set_visualize(self, visualize, show_backprop=False):
        self.visualize = visualize
        self.net.visualize = visualize
        self.net.show_backprop = show_backprop

    def set_data_settings(self, data_settings):

        self.ds = data_settings
        self.input_manager = InputManager(self.ds, self.net.input_dims)

    def set_frame(self, frame):
        self.visualizer.set_frame(frame)

    def set_box(self, min_max_box, margin_ratio):
        self.input_manager.set_box(min_max_box, margin_ratio)

    def set_train_path(self, path):
        self.train_path = path

    def train_each_case(self, tbp, data_name_list):
        data_dic = self.input_manager.get_data_dic(self.train_path)
        dist_dic = {}
        for case in data_dic:
            print "train case", case
            if tbp:
                dist_dic[case] = self.net.train(data_dic[case])
            else:
                dist_dic[case] = self.net.train_without_tbp(data_dic[case])
        return dist_dic

    def train_all(self):
        data_list = self.input_manager.get_data_all(self.train_path)
        return self.net.train(data_list)

    def test_clutter(self, path, data_path, name, tbp, case_list, single_test):

        dist_path = path + '/distribution/'
        result_path = path + '/result/'
        result_xyz = {}
        result = {}
        fail_count = {}
        distribution_dic = {}

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

    def cross_validation(self, path, name, train, tbp, data_name_list):
        self.visualize = False

        data_name_dic = self.input_manager.get_data_name_dic(self.train_path, data_name_list)
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
                        full_distribution[case] = self.net.train(train_data_list_full)
                    else:
                        full_distribution[case] = self.net.train_without_tbp(train_data_list_full)
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
                        distribution = self.net.train(train_data_list[case][test_object])
                    else:
                        distribution = self.net.train_without_tbp(train_data_list[case][test_object])

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
            filter_xyz_dict, filter_xy_dict, filter_resp_dict = self.net.get_all_filter_xyz(data, distribution)

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


    def filter_distribution(self, dist, low_n):

        if low_n < 0:
            return dist

        if self.ds.filter_same_parent:
            return self.filter_distribution_same_parent(dist, low_n)
        else:
            return self.filter_distribution_variance(dist, low_n)

    # filter out high variance features
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

    # find low variance features that has the same parent filter
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
        # print resp_dict

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


    def show_feature(self, filter_xyz_dict, ns_prefix=""):
        color_map = {}
        color_map[1] = (1,1,0)
        color_map[2] = (0,1,1)
        color_map[3] = (1,0,1)
        color_map[4] = (1,0.5,0.5)

        for sig in filter_xyz_dict:
            print sig, filter_xyz_dict[sig]

        idx = 0
        for sig in filter_xyz_dict:
            ns = ns_prefix + "/" + "/".join([str(c) for c in sig])
            self.visualizer.publish_point_array([filter_xyz_dict[sig]], idx, ns, 'feature', color_map[len(sig)], Marker.POINTS, 0.9, 0.01 )

    # add offset from feature xyz to grasp_points
    def get_distribution_cameraframe(self, dist, filter_xyz_dict):

        dist_cf = copy.deepcopy(dist.data_dict)
        for sig in dist.data_dict:
            for frame in dist.data_dict[sig]:
                if sig in filter_xyz_dict:
                    dist_cf[sig][frame] =  np.array(dist_cf[sig][frame]) + np.array(filter_xyz_dict[sig])
                    # dist_cf[sig][frame] += filter_xyz_dict[sig]
                else:
                    if sig in dist_cf:
                        dist_cf.pop(sig)
        return dist_cf


    def get_frame_xyz(self, data, frame_name):
        return data.pose_dict[frame_name][0]

    def gen_receptive_grid(self, receptive_field_size):
        return np.mgrid[0:receptive_field_size,0:receptive_field_size]

    def get_state(self, dist, data, idx=0):
        xyz_dict, xy_dict, response_dict = self.net.get_all_filter_xyz(data, dist, idx)
        return xyz_dict, xy_dict, response_dict

    # show point cloud in rviz, input server needs to be running
    def show_point_cloud(self, name):
        rospy.wait_for_service('show_point_cloud')
        try:
            show_point_cloud = rospy.ServiceProxy('show_point_cloud', String2)
            resp = show_point_cloud(name,'')
            return resp.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
