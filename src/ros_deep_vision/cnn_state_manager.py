#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from ros_deep_vision.srv import SaveData
from geometry_msgs.msg import Point, Pose
from data_util import *
from data_settings import *
from distribution import *
import time


class CNNStateManager:
    def __init__(self, settings, data_setting_case = "cnn_features"):
        ds = DataSettings(case=data_setting_case)
        ds.mask_centering = False
        self.ds = ds
        self.tbp = ds.tbp

        self.data_monster = DataMonster(settings, ds)
        self.data_monster.visualize = True
        self.path = settings.ros_dir

        self.data_monster.set_train_path(self.path + '/current/')

        self.data_collector = DataCollector(self.path + '/current/')

        self.data_monster.show_backprop = False#True#
        self.max_clusters = 3
        camera_frame = rospy.get_param('~camera_frame') # kinect_optical_frame for ubot, /r2/head/asus_depth_optical_frame for r2
        print "camera_frame", camera_frame
        self.data_monster.set_frame(camera_frame)
        # set a lower minimum box width to handle objects further away
        self.data_monster.input_manager.set_min_box_width(50)
        # the percentage of margin added to min max box
        self.box_margin = 0.5

    def set_box_param(self, min_box_width, box_margin, fix_margin):
        self.box_margin = box_margin
        self.data_monster.input_manager.set_min_box_width(min_box_width)
        self.data_monster.input_manager.set_box_fix_margin(fix_margin)
    def capture_input(self):

        time_str = strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())
        save_data = SaveDataRequest()
        save_data.name = "current_" + time_str
        save_data.action = "unknown"
        save_data.target_type = "unknown"
        save_data.result = -1
        time.sleep(0.3)

        self.data_collector.save_images(save_data.name)
        self.data_collector.save_point_cloud_multi(save_data)
        return save_data.name

    def get_box_list(self, data_name):
        box_min_max_list = []
        box_centroid_list = []
        box_f = open(self.path + '/current/' + data_name + "_box.txt", "r")
        for box_str in box_f:
            box_str_list = box_str.rstrip('\n').split(",")
            box_min_max = [int(num) for num in box_str_list[0:4]]
            box_min_max_list.append(box_min_max)
            print box_min_max
            if (len(box_str_list) > 4):
                box_centroid = [float(num) for num in box_str_list[4:7]]
                box_centroid_list.append(box_centroid)
        return box_min_max_list, box_centroid_list

    # get list of hierarchical CNN features, state_list is the expected features, finds max N if set to None
    def get_cnn_list_state(self,state_list):
        print "received grasp request"

        save_data_name = self.capture_input()
        time.sleep(0.3)
        # load crop box
        box_min_max_list, _ = self.get_box_list(save_data_name)

        # taking care of the top n largest blobs
        # box_min_max_list = box_min_max_list[0:min(self.max_clusters,len(box_min_max_list))]

        state_list_all = []
        pose_list_all = []

        box_min_max = box_min_max_list[0]

        print "handle box", box_min_max
        item_name = "item"
        self.data_monster.set_box(box_min_max, self.box_margin)
        # load image, point cloud, distribution
        data = Data()
        data.name = save_data_name
        data.img, data.mask, data.pc = None, None ,None
        while (not rospy.is_shutdown()) and (data.img is None or data.mask is None or data.pc is None):
            self.data_monster.input_manager.load_img_mask_pc(data, self.path + '/current/')
            time.sleep(0.1)

        cv2.imshow("img_"+item_name, data.img[:,:,(2,1,0)])
        cv2.imshow("mask_"+item_name, data.mask)
        cv2.imwrite(self.path + '/current/' + data.name + "_" + item_name + "_rgb_crop.png", data.img[:, :, (2,1,0)])
        cv2.waitKey(100)

        # generate grasp points
        if state_list == None:
            filter_xyz_dict, value_dict = self.data_monster.get_state(None, data)
        else:
            expected_dist = state_list_to_dist(state_list)
            # print "expected_dist", expected_dist.filter_tree
            filter_xyz_dict, value_dict = self.data_monster.get_state(expected_dist, data)


        print "show feature"
        self.data_monster.show_feature(filter_xyz_dict, item_name)

        # state_list, pose_list = to_state_pose_list(value_dict, filter_xyz_dict)
        # state_list_all += state_list
        # pose_list_all += pose_list

        return value_dict, filter_xyz_dict

        # return state_list_all, pose_list_all

    # get list of hierarchical CNN features, state_list is the expected features, finds max N if set to None
    def get_clustered_cnn_list_state(self,state_list,aspect_idx):
        print "received grasp request"

        save_data_name = self.capture_input()#"current_15-01-2017-20:38:40"#
        time.sleep(0.3)
        # load crop box
        box_min_max_list, centroid_list = self.get_box_list(save_data_name)

        # taking care of the top n largest blobs
        box_min_max_list = box_min_max_list[0:min(self.max_clusters,len(box_min_max_list))]

        if aspect_idx != None:
            box_min_max_list = [box_min_max_list[aspect_idx]]
            centroid_list = [centroid_list[aspect_idx]]

        value_dic_list = []
        xyz_dic_list = []
        img_name_list = []

        for i, box_min_max in enumerate(box_min_max_list):
            print "handle box", box_min_max
            if aspect_idx == None:
                item_name = "item"+str(i)
            else:
                item_name = "focus" + str(aspect_idx)
            self.data_monster.set_box(box_min_max, self.box_margin)
            # load image, point cloud, distribution
            data = Data()
            data.name = save_data_name
            data.img, data.mask, data.pc = None, None ,None
            while (not rospy.is_shutdown()) and (data.img is None or data.mask is None or data.pc is None):
                self.data_monster.input_manager.load_img_mask_pc_seg(data, self.path + '/current/', i)
                time.sleep(0.1)

            cv2.imshow("img_"+item_name, data.img[:,:,(2,1,0)])
            cv2.imshow("mask_"+item_name, data.mask)
            img_name = self.path + '/current/' + data.name + "_" + item_name + "_rgb.png"
            cv2.imwrite(img_name, data.img[:, :, (2,1,0)])
            img_name_list.append(img_name)

            cv2.waitKey(50)

            # generate grasp points
            if state_list == None:
                filter_xyz_dict, value_dict = self.data_monster.get_state(None, data)
            else:
                expected_dist = state_list_to_dist(state_list)
                # print "expected_dist", expected_dist.filter_tree
                filter_xyz_dict, value_dict = self.data_monster.get_state(expected_dist, data)


            print "show feature"
            self.data_monster.show_feature(filter_xyz_dict, item_name)

            value_dic_list.append(value_dict)
            xyz_dic_list.append(filter_xyz_dict)
            # state_list, pose_list = to_state_pose_list(value_dict, filter_xyz_dict)
            # clustered_state_list.append(state_list)
            # clustered_pose_list.append(pose_list)

        return value_dic_list, xyz_dic_list, centroid_list, img_name_list

if __name__ == '__main__':
    rospy.init_node('cnn_state_manager', anonymous=True)
    cnn_state_manager = CNNStateManager(settings)
    while not rospy.is_shutdown():
        key = raw_input("enter r to run once, q to quit:\n")
        if key == 'r':
            # cnn_state_manager.set_box_param(200,0.,15)
            cnn_state_manager.get_clustered_cnn_list_state(None)
        elif key == 'q':
            break
    # rospy.spin()
