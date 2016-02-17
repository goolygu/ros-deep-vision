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
import re
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

        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=100)
        rospy.init_node('data_monster')

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

    def set_path(self, path):
        self.path = path

    def train(self):

        data_list = self.get_data_all(self.path)

        # data_list = data_list[0:2]

        img_list, mask_list = self.load_img_mask(data_list, self.path)

        conv5_list = self.load_conv5(img_list, mask_list)

        # print "conv5", conv5_list.shape

        filter_idx_list = self.find_consistent_filters(conv5_list, 20)
        print filter_idx_list[0:10]
        filter_idx_list = filter_idx_list[0:3]

        distribution = Distribution()

        # handle conv 4 filters
        for filter_idx in filter_idx_list:
            print "handling filter", filter_idx
            conv4_list = self.load_conv4_fix_conv5(img_list, mask_list, filter_idx)
            filter_idx_list_conv4 = self.find_consistent_filters(conv4_list, 10)
            print filter_idx_list_conv4
            filter_idx_list_conv4 = filter_idx_list_conv4[0:3]
            frame_list_conv4 = ["r2/left_thumb_tip","r2/left_index_tip"]
            dist_list = self.gen_distribution(filter_idx_list_conv4, data_list, conv4_list, frame_list_conv4)

            for frame_idx, frame in enumerate(frame_list_conv4):
                for i, filter_idx in enumerate(filter_idx_list):
                    distribution.set('conv4',filter_idx,frame,dist_list[frame_idx,i,:,:])
        # with open(self.path + "/distribution/" + 'conv4_finger_tips' + '.yaml', 'w') as outfile:
        #     outfile.write( yaml.dump(distribution) )

        frame_list_conv5 = ["r2/left_palm"]
        dist_list = self.gen_distribution(filter_idx_list, data_list, conv5_list, frame_list_conv5)
        for frame_idx, frame in enumerate(frame_list_conv5):
            for i, filter_idx in enumerate(filter_idx_list):
                distribution.set('conv5',filter_idx,frame,dist_list[frame_idx,i,:,:])

        # distribution.save(self.path)
        return distribution

    def test(self, distribution):
        data_list = self.get_data_all(self.path)
        data_list = data_list[12:13]
        img_list, mask_list = self.load_img_mask(data_list, self.path)

        # 280/13
        resize_ratio = get_after_crop_size()[0] / self.net.blobs['conv5'].data.shape[2]
        receptive_field_size = int(round(resize_ratio))
        receptive_grid = self.gen_receptive_grid(receptive_field_size)




        for idx, data in enumerate(data_list):

            filter_xyz_dict = self.get_all_filter_xyz(data, distribution, img_list[idx], mask_list[idx])
            distribution_cf = self.get_distribution_cameraframe(distribution, filter_xyz_dict)
            self.show_point_cloud(data.name)
            self.show_distribution(distribution_cf)

            # max_xy_list = []
            # layer = 'conv5'
            # frame = "r2/left_palm"
            # self.net_proc_forward_layer(img_list[idx], mask_list[idx])
            # for filter_idx in distribution.data_dict[layer]:
            #     max_xy = self.get_filter_max_xy(self.net.blobs[layer].data[0,filter_idx])
            #     orig_xy = self.get_orig_xy(max_xy, resize_ratio)
            #     max_xy_list.append(orig_xy)
            # max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, data.name, max_xy_list, receptive_grid)
            #
            # point_list_conv5 = np.array([]).reshape([0,3])
            # for i, filter_idx in enumerate(distribution.data_dict[layer]):
            #     xyz = max_xyz_list[i]
            #     print "xyz", xyz
            #     print "diff", distribution.data_dict[layer][filter_idx][frame]
            #     points = np.array(distribution.data_dict[layer][filter_idx][frame]) + xyz
            #     point_list_conv5 = np.concatenate((point_list_conv5, points), axis = 0)
            #
            # frame_list = ["r2/left_thumb_tip","r2/left_index_tip"]
            # layer = 'conv4'
            # max_xy_list = []
            # for filter_idx in distribution.data_dict[layer]:
            #     self.net_proc_backward(filter_idx,'conv5')
            #     max_xy = self.get_filter_max_xy(self.net.blobs[layer].diff[0,filter_idx])
            #     orig_xy = self.get_orig_xy(max_xy, resize_ratio)
            #     max_xy_list.append(orig_xy)
            # max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, data.name, max_xy_list, receptive_grid)
            #
            # point_list_dict =  {}
            # for frame in frame_list:
            #     point_list_conv4 = np.array([]).reshape([0,3])
            #     for i, filter_idx in enumerate(distribution.data_dict[layer]):
            #         xyz = max_xyz_list[i]
            #         # print "xyz", xyz
            #         # print "diff", distribution.data_dict[layer][filter_idx][frame]
            #         points = np.array(distribution.data_dict[layer][filter_idx][frame]) + xyz
            #         point_list_conv4 = np.concatenate((point_list_conv4, points), axis = 0)
            #     point_list_dict[frame] = point_list_conv4
            #
            # self.show_point_cloud(data.name)
            # while not rospy.is_shutdown():
            #     self.publish_point_list(point_list_conv5, (1,0,0), 1)
            #     self.publish_point_list(point_list_dict["r2/left_thumb_tip"], (0,1,0), 2)
            #     self.publish_point_list(point_list_dict["r2/left_index_tip"], (0,0,1), 3)
            # rospy.spin()
            # new_pc = self.append_point_cloud(self.path, data.name,point_list)
            # print point_list
            # plot_dist_camera(point_list,0.9)
    def show_distribution(self, dist_cf):
        color_map = {}
        color_map["r2/left_palm"] = (1,0,0)
        color_map["r2/left_thumb_tip"] = (0,1,0)
        color_map["r2/left_index_tip"] = (0,0,1)
        while not rospy.is_shutdown():
            id = 0
            for layer in dist_cf:
                for filter_idx in dist_cf[layer]:
                    for frame in dist_cf[layer][filter_idx]:
                        self.publish_point_list(dist_cf[layer][filter_idx][frame], color_map[frame], id)
                        id += 1

    def get_distribution_cameraframe(self, dist, filter_xyz_dict):
        dist_cf = copy.deepcopy(dist.data_dict)
        for layer in dist_cf:
            for filter_idx in dist_cf[layer]:
                for frame in dist_cf[layer][filter_idx]:
                    dist_cf[layer][filter_idx][frame] += filter_xyz_dict[layer][filter_idx]
        return dist_cf

    def get_all_filter_xyz(self, data, dist, img, mask):
        xyz_dict = {}
        for layer in dist.data_dict:
            max_xy_list = []
            xyz_dict[layer] = {}

            resize_ratio = get_after_crop_size()[0] / self.net.blobs[layer].data.shape[2]
            receptive_field_size = int(round(resize_ratio))
            receptive_grid = self.gen_receptive_grid(receptive_field_size)

            self.net_proc_forward_layer(img, mask)
            for filter_idx in distribution.data_dict[layer]:
                max_xy = self.get_filter_max_xy(self.net.blobs[layer].data[0,filter_idx])
                orig_xy = self.get_orig_xy(max_xy, resize_ratio)
                max_xy_list.append(orig_xy)
            max_xyz_list = self.get_average_xyz_from_point_cloud(self.path, data.name, max_xy_list, receptive_grid)

            for i, filter_idx in enumerate(distribution.data_dict[layer]):
                xyz = max_xyz_list[i]
                xyz_dict[layer][filter_idx] = xyz

        return xyz_dict


    def find_consistent_filters(self, conv_list, threhold):
        hist = np.zeros(conv_list.shape[1])
        max_hist = np.zeros(conv_list.shape[1])
        for idx, conv in enumerate(conv_list):
            # print idx
            bin_data, max_data = self.binarize(conv, threhold)
            hist = hist + bin_data
            max_hist = np.amax(np.concatenate((max_hist[np.newaxis,...],max_data[np.newaxis,...]),axis=0),axis=0)
        print "hist", hist
        print "max hist", max_hist
        filter_idx_list = np.argsort(hist)[::-1]
        return filter_idx_list

    def get_filter_max_xy(self, filter_response):
        assert filter_response.ndim == 2, "filter size incorrect"
        max_value = np.amax(filter_response)
        if max_value <= 0:
            return np.array([float('nan'),float('nan')])
        max_idx = np.argmax(filter_response, axis=None)
        max_xy = np.unravel_index(max_idx, filter_response.shape)
        return max_xy

    def gen_distribution(self, filter_idx_list, data_list, conv_list, frame_list):

        dist_list = np.empty([len(frame_list),len(filter_idx_list),len(data_list),3])
        # 280/13
        resize_ratio = get_after_crop_size()[0] / conv_list.shape[2]
        receptive_field_size = int(round(resize_ratio))
        receptive_grid = self.gen_receptive_grid(receptive_field_size)

        for idx, conv in enumerate(conv_list):
            print idx
            data = data_list[idx]
            name = data.name
            max_idx_list = []
            max_xy_list = []
            for filter_idx in filter_idx_list:
                max_xy = self.get_filter_max_xy(conv[filter_idx])
                # max_value = np.amax(conv[filter_idx])
                # if max_value <= 0:
                #     max_xy_list.append(np.array([float('nan'),float('nan')]))
                #     continue
                #
                # max_idx = np.argmax(conv[filter_idx], axis=None)
                # # print "max idx", max_idx
                # max_xy = np.unravel_index(max_idx, conv.shape[1:3])
                # print "max", max_xy
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

        for idx in range(len(self.available_layer)-1):
            output = self.net.forward(data=data_blob,start=self.available_layer[idx],end=self.available_layer[idx+1])
            if self.available_layer[idx].startswith("conv"):
                new_blob = self.net.blobs[self.available_layer[idx]].data
                new_blob.data = self.mask_out(self.net.blobs[self.available_layer[idx]].data, mask)
                self.net.blobs[self.available_layer[idx]] = new_blob
            # print output
        return output

    def net_proc_backward(self, filter_idx, backprop_layer):

        back_mode = 'grad'

        diffs = self.net.blobs[backprop_layer].diff * 0
        diffs[0][filter_idx] = self.net.blobs[backprop_layer].data[0,filter_idx]

        assert back_mode in ('grad', 'deconv')
        if back_mode == 'grad':
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

    def get_average_xyz_from_point_cloud(self, path, name, max_xy_list, receptive_grid):
        p = pcl.PointCloud()
        p.from_file(path + name + ".pcd")
        a = np.asarray(p)
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
            avg = np.nanmean(a[idx_receptive_list],axis=0)
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

    def publish_point_list(self, point_list, color, id):

        pl_marker = Marker()
        pl_marker.header.frame_id = "/r2/asus_frame"
        pl_marker.header.stamp = rospy.Time()
        pl_marker.id = id
        pl_marker.type = Marker.POINTS
        # pl_marker.pose.position.x = 1
        # pl_marker.pose.position.y = 1
        # pl_marker.pose.position.z = 1
        pl_marker.scale.x = 0.02
        pl_marker.scale.y = 0.02
        pl_marker.scale.z = 0.02
        pl_marker.color.a = 1
        pl_marker.color.r = color[0]
        pl_marker.color.g = color[1]
        pl_marker.color.b = color[2]
        pl_marker.lifetime = rospy.Duration.from_sec(60)
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


    def get_data_all(self,path):
        data_file_list = []
        match_flags = re.IGNORECASE
        for filename in os.listdir(path):
            if re.match('.*_data\.yaml$', filename, match_flags):
                data_file_list.append(filename)
        data_file_list = sorted(data_file_list)

        # data_dict = {}
        data_list = []

        for data_file in data_file_list:
            f = open(path+data_file)
            data = yaml.load(f)
            data_list.append(data)
            # data_dict[data.name] = data
        return data_list

    def load_img_mask(self, data_list, path):

        img_list = []
        mask_list = []

        for idx, data in enumerate(data_list):
            # print idx

            img_name = path + data.name + "_rgb.png"
            img = cv2_read_file_rgb(img_name)
            img = crop_to_center(img)
            img = cv2.resize(img, self.input_dims)
            img_list.append(img)

            mask_name = path + data.name + "_mask.png"
            mask = cv2.imread(mask_name)
            mask = crop_to_center(mask)
            mask = np.reshape(mask[:,:,0], (mask.shape[0], mask.shape[1]))
            mask_list.append(mask)

        return img_list, mask_list


    def load_conv5(self, img_list, mask_list):

        conv5_list = np.array([]).reshape([0] + list(self.net.blobs['conv5'].data.shape[1:]))

        for idx, img in enumerate(img_list):
            print idx
            # print "img", img.shape
            self.net_proc_forward_layer(img, mask_list[idx])
            # self.net_preproc_forward(img)
            conv5_list = np.append(conv5_list, self.net.blobs['conv5'].data, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv5_list

    def load_conv4_fix_conv5(self, img_list, mask_list, filter_idx):

        conv4_list = np.array([]).reshape([0] + list(self.net.blobs['conv4'].data.shape[1:]))

        for idx, img in enumerate(img_list):
            print idx
            # print "img", img.shape
            self.net_proc_forward_layer(img, mask_list[idx])
            self.net_proc_backward(filter_idx,'conv5')

            conv4_list = np.append(conv4_list, self.net.blobs['conv4'].diff, axis=0)
            # print "shape", self.net.blobs['conv5'].data.shape
        return conv4_list

    def binarize(self, data, threhold):
        # print data.shape
        bin_data = np.zeros(data.shape[0])
        max_data = np.zeros(data.shape[0])
        for id, filter in enumerate(data):
            max_value = np.amax(filter)
            if max_value > threhold:
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

    data_monster = DataMonster(settings)
    data_monster.set_path(settings.ros_dir + '/data/')

    # distribution = data_monster.train()
    # distribution.save(settings.ros_dir + '/data/', 'conv5_palm_conv4_finger')
    distribution = Distribution()
    distribution.load(settings.ros_dir + '/data/', 'conv5_palm_conv4_finger')

    data_monster.test(distribution)
