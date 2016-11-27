#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from perception_msgs.srv import GetGraspPoints, SaveData
from geometry_msgs.msg import Point
from data_util import *
from data_settings import *
import time
import cv2

class GraspGenerator:
    def __init__(self, settings):
        rospy.init_node('grasp_generator', anonymous=True)
        ds = DataSettings()
        self.tbp = ds.tbp

        self.data_monster = DataMonster(settings, ds)
        self.data_monster.visualize = True
        self.dist_path = settings.ros_dir + '/distribution/'
        self.current_path = settings.ros_dir + '/current/'

        # self.data_monster.set_train_path()
        asus_only = True
        self.data_collector = DataCollector(self.current_path, asus_only)

        dist_name = ds.get_name()#'(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
        self.data_monster.show_backprop = True#False#
        self.data_monster.input_manager.set_visualize(True)
        self.distribution = Distribution()
        case1 = '[side_wrap:cylinder]'
        case2 = '[side_wrap:cuboid]'
        self.distribution.load(self.dist_path + "cross_validation/", case2 + '[leave_tazobox]' + dist_name)
        # self.distribution.load(self.dist_path + "cross_validation/", case1 + '[leave_cjar]' + dist_name)
        if ds.filter_low_n > 0:
            self.distribution = self.data_monster.filter_distribution(self.distribution, ds.filter_low_n)
            # self.distribution = self.data_monster.filter_distribution_same_parent(self.distribution, ds.filter_low_n)
        s = rospy.Service('get_grasp_points', GetGraspPoints, self.handle_get_grasp_points)#_multi)

    def handle_get_grasp_points(self,req):
        print "received grasp request"
        cv2.destroyAllWindows()
        # TODO call data_collector service instead, run in seperate thread to save time
        # first save image and point cloud
        time_str = strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())
        save_data = SaveDataRequest()
        save_data.name = "current_" + time_str
        save_data.action = "unknown"
        save_data.target_type = "unknown"
        save_data.result = -1

        self.data_collector.save_images(save_data.name)
        self.data_collector.save_point_cloud_current(save_data)

        time.sleep(0.3)

        # load image, point cloud, distribution
        data = Data()
        data.name = save_data.name
        data.img, data.mask = None, None
        while data.img is None and data.mask is None:
            self.data_monster.input_manager.load_img_mask_pc(data, self.current_path)
            time.sleep(0.1)

        # generate grasp points

        if self.tbp:
            filter_xyz_dict, filter_resp_dict = self.data_monster.get_all_filter_xyz(data, self.distribution)
        else:
            filter_xyz_dict, filter_resp_dict = self.data_monster.get_all_filter_xyz_notbp(data, self.distribution)
        # filter_xyz_dict = self.data_monster.get_all_filter_xyz(data, self.distribution, img_list[0], mask_list[0])

        distribution_cf = self.data_monster.get_distribution_cameraframe(self.distribution, filter_xyz_dict)
        self.data_monster.show_point_cloud(data.name)
        avg_dic = self.data_monster.model_distribution(distribution_cf, filter_resp_dict)

        self.data_monster.show_distribution(distribution_cf)

        resp = GetGraspPointsResponse()
        frame_list = []
        point_list = []
        for frame in avg_dic:
            frame_list.append(frame)
            point = Point()
            point.x = avg_dic[frame][0]
            point.y = avg_dic[frame][1]
            point.z = avg_dic[frame][2]
            point_list.append(point)

        resp.names = tuple(frame_list)
        resp.positions = tuple(point_list)

        return resp

    def handle_get_grasp_points_multi(self,req):
        print "received grasp request"

        # TODO call data_collector service instead, run in seperate thread to save time
        # first save image and point cloud
        time_str = strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())
        save_data = SaveDataRequest()
        save_data.name = "current_" + time_str
        save_data.action = "unknown"
        save_data.target_type = "unknown"
        save_data.result = -1

        self.data_collector.save_images(save_data.name)
        self.data_collector.save_point_cloud_multi(save_data)

        time.sleep(0.3)

        # load crop box

        box_min_max_list = []

        box_f = open(self.current_path + save_data.name + "_box.txt", "r")
        for box_str in box_f:
            box_min_max = box_str.rstrip('\n').split(",")
            box_min_max = [int(num) for num in box_min_max]
            box_min_max_list.append(box_min_max)
            print box_min_max


        for box_min_max in box_min_max_list:
            self.data_monster.set_box(box_min_max, 0.7)
            # load image, point cloud, distribution
            data = Data()
            data.name = save_data.name
            data.img, data.mask = None, None
            while data.img is None and data.mask is None:
                self.data_monster.input_manager.load_img_mask_pc(data, self.current_path)
                time.sleep(0.1)

            cv2.imshow("img", data.img)
            cv2.waitKey(100)

            # generate grasp points

            if self.tbp:
                filter_xyz_dict, filter_resp_dict = self.data_monster.get_all_filter_xyz(data, self.distribution)
            else:
                filter_xyz_dict, filter_resp_dict = self.data_monster.get_all_filter_xyz_notbp(data, self.distribution)
            # filter_xyz_dict = self.data_monster.get_all_filter_xyz(data, self.distribution, img_list[0], mask_list[0])
            self.data_monster.show_feature(filter_xyz_dict)

            distribution_cf = self.data_monster.get_distribution_cameraframe(self.distribution, filter_xyz_dict)
            self.data_monster.show_point_cloud(data.name)
            avg_dic = self.data_monster.model_distribution(distribution_cf, filter_resp_dict)

            self.data_monster.show_distribution(distribution_cf)

            resp = GetGraspPointsResponse()
            frame_list = []
            point_list = []
            for frame in avg_dic:
                frame_list.append(frame)
                point = Point()
                point.x = avg_dic[frame][0]
                point.y = avg_dic[frame][1]
                point.z = avg_dic[frame][2]
                point_list.append(point)

            resp.names = tuple(frame_list)
            resp.positions = tuple(point_list)

            return resp


if __name__ == '__main__':
    grasp_generator = GraspGenerator(settings)

    # grasp_generator.handle_get_grasp_points(None)
    rospy.spin()
