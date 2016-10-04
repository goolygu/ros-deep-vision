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
from core import CodependentThread
from image_misc import *

from time import gmtime, strftime
from utils import DescriptorHandler, Descriptor
import settings
import caffe
import re
import rospkg

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ros_deep_vision.srv import *
from geometry_msgs.msg import *
from visualization_msgs.msg import *
import yaml
import threading

import tf

class Data:
    def __init__(self):
        self.name = ""
        self.pose_dict = {}
        self.action = ""
        self.target_type = ""
        self.result = 0
        pass

    def set(self, name, pose_dict, action, target_type, result):
        self.name = name
        self.pose_dict = pose_dict
        self.action = action
        self.target_type = target_type
        self.result = result


class DataCollector:
    def __init__(self, path):

        self.bridge = CvBridge()
        self.camera_frame = "/r2/head/asus_depth_optical_frame"

        self.lock = threading.Lock()
        self.path = path

        self.rgb_sub = rospy.Subscriber("/asus/rgb/image_raw",Image,self.rgb_callback,queue_size=1)
        self.depth_sub = rospy.Subscriber("/asus/depth/image_raw",Image,self.depth_callback,queue_size=1)


        self.mask_sub = rospy.Subscriber("/image_mask",Image, self.mask_callback, queue_size=1)
        self.listener = tf.TransformListener()

        self.frame_list = ["r2/left_palm", "r2/left_index_base", "r2/left_index_yaw", "r2/left_index_proximal", \
                           "r2/left_index_medial", "r2/left_index_distal", "r2/left_index_tip", "r2/left_middle_base", "r2/left_middle_yaw", \
                           "r2/left_middle_proximal", "r2/left_middle_medial", "r2/left_middle_distal", "r2/left_middle_tip", "r2/left_little_proximal", \
                           "r2/left_little_tip", "r2/left_ring_proximal", "r2/left_ring_tip", "r2/left_thumb_base", \
                           "r2/left_thumb_proximal", "r2/left_thumb_medial_prime", "r2/left_thumb_medial", "r2/left_thumb_distal", \
                           "r2/left_thumb_tip"]

        s = rospy.Service('save_data', SaveData, self.handle_save_data)


    def save_point_cloud(self, req):
        rospy.wait_for_service('save_point_cloud')
        try:
            save_point_cloud = rospy.ServiceProxy('save_point_cloud', SaveData)
            resp = save_point_cloud(req)
            print "point cloud saved"
            return resp.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def save_point_cloud_current(self, req):
        rospy.wait_for_service('save_point_cloud_current')
        try:
            save_point_cloud = rospy.ServiceProxy('save_point_cloud_current', SaveData)
            resp = save_point_cloud(req)
            return resp.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def save_point_cloud_multi(self, req):
        rospy.wait_for_service('save_point_cloud_multi')
        try:
            save_point_cloud = rospy.ServiceProxy('save_point_cloud_multi', SaveData)
            resp = save_point_cloud(req)
            return resp.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def rgb_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            with self.lock:
                self.rgb_image = cv_image
        except CvBridgeError, e:
            print e

    def depth_callback(self,data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            with self.lock:
                self.depth_image = cv_depth
        except CvBridgeError, e:
            print e

    def mask_callback(self,data):
        try:
            mask_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            with self.lock:
                self.mask_image = mask_image
        except CvBridgeError, e:
            print e

    # def pose_callback(self, data):
    #     with self.lock:
    #         self.ee_pose = data.pose

    def save_images(self, name):
        base_name = self.path + name
        with self.lock:
            cv2.imwrite(base_name + "_rgb.png", self.rgb_image)
            cv2.imwrite(base_name + "_depth.png", self.depth_image)
            cv2.imwrite(base_name + "_mask.png", self.mask_image)
            # cv2.imwrite(base_name + "_rgb_crop.png", crop_to_center(self.rgb_image))

    def get_pose(self, base_frame, end_frame):
        (trans,rot) = self.listener.lookupTransform(base_frame, end_frame, rospy.Time(0))
        return (trans,rot)


    def get_pose_dict(self):
        pose_dict = {}
        for frame in self.frame_list:
            (trans,rot) = self.get_pose(self.camera_frame, frame)
            pose_dict[frame] = (trans,rot)
        return pose_dict


    def handle_save_data(self, req):
        print "saving"
        name = req.name
        if req.name == "Time":
            name = strftime("%d-%m-%Y-%H:%M:%S", gmtime())

        data = Data()
        pose_dict = self.get_pose_dict()
        with self.lock:
            data.set(name, pose_dict, req.action, req.target_type, req.result)

        with open(self.path + name + '_data.yaml', 'w') as outfile:
            outfile.write( yaml.dump(data) )

        self.save_images(name)
        self.save_point_cloud(req)
        resp = SaveDataResponse()
        resp.result = 0
        return resp

if __name__ == '__main__':
    rospy.init_node('data_collector', anonymous=False)
    rospack = rospkg.RosPack()
    ros_dir = rospack.get_path('ros_deep_vision')
    data_collector = DataCollector(ros_dir + "/data/set0/", True)

    rospy.spin()
