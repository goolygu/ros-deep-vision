#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from perception_msgs.srv import GetGraspPoints, SaveData, GetListState
from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose
from data_util import *
from data_settings_atg import *
import time

class CNNStateManager:
    def __init__(self, settings):
        rospy.init_node('cnn_state_manager', anonymous=True)
        self.tbp = True#False#
        ds = DataSettings(self.tbp)
        self.data_monster = DataMonster(settings, ds)
        self.data_monster.visualize = True
        self.path = settings.ros_dir + '/data/'
        if self.tbp:
            self.data_path = settings.ros_dir + '/data/'
        else:
            self.data_path = settings.ros_dir + '/data_notbp/'

        self.data_monster.set_path(self.path + 'current/')
        self.data_collector = DataCollector(self.path + 'current/')

        dist_name = ds.get_name()#'(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
        self.data_monster.show_backprop = True#False#
        # self.distribution = Distribution()
        # case1 = '[side_wrap:cylinder]'
        # case2 = '[side_wrap:cuboid]'
        # self.distribution.load(self.data_path, case1 + dist_name)

        s = rospy.Service('get_cnn_state', GetListState, self.handle_get_cnn_state)


    def handle_get_cnn_state(self,req):
        print "received grasp request"

        # TODO call data_collector service instead, run in seperate thread to save time
        # first save image and point cloud
        time_str = strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())
        save_data = SaveDataRequest()
        save_data.name = "current_" + time_str
        save_data.action = "unknown"
        save_data.target_type = "unknown"
        save_data.result = -1
        time.sleep(0.3)

        self.data_collector.save_images(save_data.name)
        self.data_collector.save_point_cloud_multi(save_data)

        time.sleep(0.3)

        # load crop box

        box_min_max_list = []

        box_f = open(self.path + 'current/' + save_data.name + "_box.txt", "r")
        for box_str in box_f:
            box_min_max = box_str.rstrip('\n').split(",")
            box_min_max = [int(num) for num in box_min_max]
            box_min_max_list.append(box_min_max)
            print box_min_max


        for box_min_max in box_min_max_list:
            self.data_monster.set_box(box_min_max, 0.5)
            # load image, point cloud, distribution
            data = Data()
            data.name = save_data.name
            img_list, mask_list = None, None
            while img_list is None and mask_list is None:
                img_list, mask_list = self.data_monster.load_img_mask([data], self.path + 'current/')
                time.sleep(0.1)

            print "img size", img_list[0].shape
            print "mask size", mask_list[0].shape

            cv2.imshow("img", img_list[0])
            cv2.imshow("mask", mask_list[0])
            cv2.imwrite(self.path + 'current/' + save_data.name + "_rgb_crop.png", img_list[0][:, :, (2,1,0)])
            cv2.waitKey(100)

            # generate grasp points

            filter_xyz_dict = self.data_monster.get_state(data.name, img_list[0], mask_list[0])
            # filter_xyz_dict = self.data_monster.get_all_filter_xyz(data, self.distribution, img_list[0], mask_list[0])
            self.data_monster.show_feature(filter_xyz_dict)

            resp = GetListStateResponse()
            state_list = []
            pose_list = []
            for sig in filter_xyz_dict:
                state = State()
                state.type = 'cnn'
                state.name = str(sig)
                state.value = 1
                state_list.append(state)
                pose = Pose()
                pose.position.x = filter_xyz_dict[sig][0]
                pose.position.y = filter_xyz_dict[sig][1]
                pose.position.z = filter_xyz_dict[sig][2]
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1
                pose_list.append(pose)

            resp.state_list = tuple(state_list)
            resp.pose_list = tuple(pose_list)

            return resp



if __name__ == '__main__':
    cnn_state_manager = CNNStateManager(settings)
    cnn_state_manager.handle_get_cnn_state(None)
    rospy.spin()
