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
    def __init__(self, settings):
        rospy.init_node('cnn_state_manager', anonymous=True)
        ds = DataSettings(case="cnn_features")
        ds.mask_centering = False
        self.tbp = ds.tbp

        self.data_monster = DataMonster(settings, ds)
        self.data_monster.visualize = True
        self.path = settings.ros_dir

        self.data_monster.set_train_path(self.path + '/current/')

        self.data_collector = DataCollector(self.path + '/current/')

        self.data_monster.show_backprop = False#True#
        self.max_clusters = 3
        camera_frame = rospy.get_param('~camera_frame') # kinect_optical_frame for ubot
        print "camera_frame", camera_frame
        self.data_monster.set_frame(camera_frame)
        # set a lower minimum box width to handle objects further away
        self.data_monster.input_manager.set_min_box_width(50)

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

        box_f = open(self.path + '/current/' + data_name + "_box.txt", "r")
        for box_str in box_f:
            box_min_max = box_str.rstrip('\n').split(",")
            box_min_max = [int(num) for num in box_min_max]
            box_min_max_list.append(box_min_max)
            print box_min_max
        return box_min_max_list

    # get list of hierarchical CNN features, state_list is the expected features, finds max N if set to None
    def get_cnn_list_state(self,state_list):
        print "received grasp request"

        save_data_name = self.capture_input()
        time.sleep(0.3)
        # load crop box
        box_min_max_list = self.get_box_list(save_data_name)

        # taking care of the top n largest blobs
        box_min_max_list = box_min_max_list[0:min(self.max_clusters,len(box_min_max_list))]

        state_list_all = []
        pose_list_all = []

        for i, box_min_max in enumerate(box_min_max_list):
            print "handle box", box_min_max
            item_name = "item_"+str(i)
            self.data_monster.set_box(box_min_max, 0.5)
            # load image, point cloud, distribution
            data = Data()
            data.name = save_data_name
            data.img, data.mask, data.pc = None, None ,None
            while data.img is None or data.mask is None or data.pc is None:
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

            state_list, pose_list = to_state_pose_list(value_dict, filter_xyz_dict)
            state_list_all.append(state_list)
            pose_list_all.append(pose_list)

        return state_list_all, pose_list_all



if __name__ == '__main__':
    cnn_state_manager = CNNStateManager(settings)
    while not rospy.is_shutdown():
        key = raw_input("enter r to run once, q to quit:\n")
        if key == 'r':
            cnn_state_manager.get_cnn_list_state(None)
        elif key == 'q':
            break
    # rospy.spin()
