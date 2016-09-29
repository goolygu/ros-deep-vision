#! /usr/bin/env python
import math
import cv2
import pcl
import numpy as np
import yaml
import re
import os
from data_collector import Data
from image_misc import *
import sys


class InputManager:

    def __init__(self, ds, input_dims):
        self.ds = ds
        self.input_dims = input_dims
        self.frame_x = 480
        self.frame_y = 640
        self.min_box = 300
        # self.set_box([200,460,180,440], 0)
        self.set_width(self.ds.input_width)
        self.point_cloud_shape = (480,640)
        self.visualize = False

    def set_width(self, width):
        self.set_box([200,200+width,180,180+width], 0)

    def set_visualize(self, vis):
        self.visualize = vis

    def set_center(self, center_xy):
        curr_center = [0,0]
        curr_center[0] = (self.min_max_box[0] + self.min_max_box[1])/2
        curr_center[1] = (self.min_max_box[2] + self.min_max_box[3])/2
        off_set = [0,0]
        off_set[0] = center_xy[0] - curr_center[0]
        off_set[1] = center_xy[1] - curr_center[1]

        if self.min_max_box[0] + off_set[0] < 0:
            off_set[0] = 0 - self.min_max_box[0]
        if self.min_max_box[1] + off_set[0] >= self.frame_x:
            off_set[0] = self.frame_x -self.min_max_box[1]
        if self.min_max_box[2] + off_set[1] < 0:
            off_set[1] = 0 - self.min_max_box[2]
        if self.min_max_box[3] + off_set[1] >= self.frame_y:
            off_set[1] = self.frame_y -self.min_max_box[3]

        self.min_max_box[0] += off_set[0]
        self.min_max_box[1] += off_set[0]
        self.min_max_box[2] += off_set[1]
        self.min_max_box[3] += off_set[1]

    def set_box(self, min_max_box, margin_ratio):
        self.min_max_box_orig = min_max_box
        min_x_orig = min_max_box[0]
        max_x_orig = min_max_box[1]
        min_y_orig = min_max_box[2]
        max_y_orig = min_max_box[3]
        self.width = max(max_x_orig-min_x_orig, max_y_orig-min_y_orig)
        self.margin = round(self.width * margin_ratio)
        if self.width + 2*self.margin > self.frame_x:
            self.margin = int((self.frame_x - self.width)/2.0)

        if self.margin < 0:
            center_y = (min_y_orig + max_y_orig)/2
            min_x = 0
            max_x = self.frame_x - 1
            min_y = center_y - self.frame_x/2
            max_y = center_y + self.frame_x/2
            self.min_max_box = [min_x, max_x, min_y, max_y]
            return
        min_x = min_x_orig - self.margin
        max_x = min_x_orig + self.width + self.margin
        min_y = min_y_orig - self.margin
        max_y = min_y_orig + self.width + self.margin

        if (max_x - min_x) < self.min_box:
            min_x -= (self.min_box - (max_x - min_x))/2
            max_x += (self.min_box - (max_x - min_x))/2
            min_y -= (self.min_box - (max_y - min_y))/2
            max_y += (self.min_box - (max_y - min_y))/2

        # consider corner conditions
        if max_x > self.frame_x:
            shift = max_x - self.frame_x
            min_x -= shift
            max_x = self.frame_x
        if min_x < 0:
            shift = -min_x
            max_x += shift
            min_x = 0

        if max_y > self.frame_y:
            shift = max_y - self.frame_y
            min_y -= shift
            max_y = self.frame_y
        if min_y < 0:
            shift = -min_y
            max_y += shift
            min_y = 0

        self.min_max_box = [min_x, max_x, min_y, max_y]
        # print "min max box", min_max_box

    def crop(self, frame):
        min_x = self.min_max_box[0]
        max_x = self.min_max_box[1]
        min_y = self.min_max_box[2]
        max_y = self.min_max_box[3]
        if len(frame.shape) > 2:
            return frame[min_x:max_x,min_y:max_y,:]
        else:
            return frame[min_x:max_x,min_y:max_y]

    def get_crop_bias(self):
        min_x = self.min_max_box[0]
        min_y = self.min_max_box[2]
        return (min_x, min_y)

    def get_after_crop_size(self):
        return (self.min_max_box[1] - self.min_max_box[0], self.min_max_box[3] - self.min_max_box[2])


        # return (self.width + 2*self.margin, self.width + 2*self.margin)


    def get_point_cloud_array(self, path, name, seg):
        p = pcl.PointCloud()
        if seg == 'seg':
            p.from_file(path + name + "_seg.pcd")
        elif seg == 'noseg':
            p.from_file(path + name + ".pcd")
        # print "width", p.width, "height", p.height, "size", p.size
        a = np.asarray(p)
        return a

    def load_img_mask_pc(self, data, path):

        mask_name = path + data.name + "_mask.png"
        mask = cv2.imread(mask_name)
        if mask is None:
            print "[ERROR] No mask"
            return None, None
        mask = np.reshape(mask[:,:,0], (mask.shape[0], mask.shape[1]))
        if not self.ds.dataset == "set1":
            center = self.get_mask_center(mask)
            self.set_center(center)

        mask = self.crop(mask)
        mask = cv2.resize(mask, self.input_dims)


        img_name = path + data.name + "_rgb.png"
        img = cv2_read_file_rgb(img_name)
        if img is None:
            print "[ERROR] No image"
            return None, None


        img = self.crop(img)
        img = cv2.resize(img, self.input_dims)

        # load point cloud
        pc_array = self.get_point_cloud_array(path, data.name, self.ds.pointcloud)
        pc = pc_array.reshape(self.point_cloud_shape + (3,))
        pc = self.crop(pc)


        if self.visualize:
            cv2.imshow("img", img)
            cv2.waitKey(100)

            cv2.imshow("mask", mask)
            cv2.waitKey(100)

            cv2.imshow("pc", pc)
            cv2.waitKey(100)

        data.img = img
        data.mask = mask
        data.pc = pc

    def get_data_by_name(self, path, name):

        f = open(path+name+"_data.yaml")
        data = yaml.load(f)
        self.load_img_mask_pc(data, path)
        return data

    # returns a dictionary with action type: target type as keys and stores a list of data
    def get_data_dic(self, path, name_list):

        data_dic = {}

        for i, data_name in enumerate(name_list):
            print i,
            sys.stdout.flush()
            data = self.get_data_by_name(path, data_name)
            key = data.action + ":" + data.target_type
            if key not in data_dic:
                data_dic[key] = []
            data_dic[key].append(data)
            # data_dict[data.name] = data
        return data_dic

    # returns a dictionary of dictionary with form dic[action:type][object_name]
    def get_data_name_dic(self, path, name_list):
        if name_list == None:
            name_list = self.get_data_name_all(path)

        data_dic = {}
        print "loading"
        for i, data_name in enumerate(name_list):
            print i,
            sys.stdout.flush()
            data = self.get_data_by_name(path, data_name)
            key = data.action + ":" + data.target_type
            object = data.name.split('_')[0]
            if key not in data_dic:
                data_dic[key] = {}
            if object not in data_dic[key]:
                data_dic[key][object] = []

            data_dic[key][object].append(data)
            # data_dict[data.name] = data
        print "finished"
        return data_dic

    # returns a list that contains all data
    def get_data_all(self, path, idx_list=None):
        print "loading"
        data_file_list = []
        match_flags = re.IGNORECASE
        for filename in os.listdir(path):
            if re.match('.*_data\.yaml$', filename, match_flags):
                data_file_list.append(filename)

        data_file_list = sorted(data_file_list)
        if idx_list != None:
            data_file_list = [ data_file_list[index] for index in idx_list]
        # print data_file_list
        data_list = []
        for i, data_file in enumerate(data_file_list):
            print i,
            sys.stdout.flush()
            f = open(path+data_file)
            data = yaml.load(f)
            self.load_img_mask_pc(data, path)
            data_list.append(data)

        return data_list

    def get_data_name_all(self, path):
        data_file_list = []
        match_flags = re.IGNORECASE
        for filename in os.listdir(path):
            if re.match('.*_data\.yaml$', filename, match_flags):
                data_file_list.append(filename)

        data_file_list = sorted(data_file_list)
        # print data_file_list
        # data_dict = {}
        name_list = []
        for data_file in data_file_list:
            f = open(path+data_file)
            data = yaml.load(f)
            name_list.append(data.name)
            # data_dict[data.name] = data
        return name_list

    def get_mask_center(self, mask):
        xy_grid = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]

        mask_norm = mask / float(np.sum(mask))
        avg_x = np.sum(xy_grid[0] * mask_norm)
        avg_y = np.sum(xy_grid[1] * mask_norm)

        return np.around(np.array([avg_x, avg_y])).astype(int)

if __name__ == '__main__':
    print "test"
    input_manager = InputManager(None,None)
    input_manager.set_width(260)
    input_manager.set_box([68,312,221,394],0.5)
    input_manager.set_center([187,306])
    print input_manager.min_max_box
