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
from distribution import *
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
        asus_only = False
        self.data_collector = DataCollector(self.path + 'current/', asus_only)

        dist_name = ds.get_name()#'(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
        self.data_monster.show_backprop = False#True#
        # self.distribution = Distribution()
        # case1 = '[side_wrap:cylinder]'
        # case2 = '[side_wrap:cuboid]'
        # self.distribution.load(self.data_path, case1 + dist_name)

        s = rospy.Service('get_cnn_list_state', GetListState, self.handle_get_cnn_list_state)

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

        box_f = open(self.path + 'current/' + data_name + "_box.txt", "r")
        for box_str in box_f:
            box_min_max = box_str.rstrip('\n').split(",")
            box_min_max = [int(num) for num in box_min_max]
            box_min_max_list.append(box_min_max)
            print box_min_max
        return box_min_max_list

    def state_list_to_dist(self, state_list):
        dist = Distribution()
        for state in state_list:
            # print "state", state.name
            dist.set_tree_feature(state.name)
        return dist


    def handle_get_cnn_list_state(self,req):
        print "received grasp request"

        save_data_name = self.capture_input()
        time.sleep(0.3)
        # load crop box
        box_min_max_list = self.get_box_list(save_data_name)

        for box_min_max in box_min_max_list:
            self.data_monster.set_box(box_min_max, 0.5)
            # load image, point cloud, distribution
            data = Data()
            data.name = save_data_name
            img, mask = None, None
            while img is None and mask is None:
                img, mask, pc = self.data_monster.load_img_mask_pc(data, self.path + 'current/')
                time.sleep(0.1)

            cv2.imshow("img", img)
            cv2.imshow("mask", mask)
            cv2.imwrite(self.path + 'current/' + save_data_name + "_rgb_crop.png", img[:, :, (2,1,0)])
            cv2.waitKey(100)

            # generate grasp points
            if req.state_list[0].name == "None":
                filter_xyz_dict, value_dict = self.data_monster.get_state(data.name, None, img, mask, pc)
            else:
                expected_dist = self.state_list_to_dist(req.state_list)
                # print "expected_dist", expected_dist.filter_tree
                filter_xyz_dict, value_dict = self.data_monster.get_state(data.name, expected_dist, img, mask, pc)


            print "show feature"
            self.data_monster.show_feature(filter_xyz_dict)

            print "form message"
            resp = GetListStateResponse()
            state_list = []
            pose_list = []
            for sig in value_dict:
                state = State()
                state.type = 'cnn'
                state.name = str(sig)
                state.value = value_dict[sig]
                state_list.append(state)

                pose = Pose()
                if not state.value == 0:
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
            print "send msg"
            return resp



if __name__ == '__main__':
    cnn_state_manager = CNNStateManager(settings)
    # cnn_state_manager.handle_get_cnn_state(None)
    rospy.spin()
