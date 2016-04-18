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

class GraspGenerator:
    def __init__(self, settings):
        rospy.init_node('grasp_generator', anonymous=True)
        ds = DataSettings()
        self.data_monster = DataMonster(settings, ds)
        self.path = settings.ros_dir + '/data/'
        self.data_monster.set_path(self.path + 'current/')
        self.data_collector = DataCollector(self.path + 'current/')

        dist_name = '(4-p-3-f)_(3-5-7)_auto_max_all_seg_103_g_bxy_5_(30-5-0.2)_above'
        self.data_monster.show_backprop = False
        self.distribution = Distribution()
        case1 = '[side_wrap:cylinder]'
        case2 = '[side_wrap:cuboid]'
        self.distribution.load(self.path, case1 + dist_name)

        s = rospy.Service('get_grasp_points', GetGraspPoints, self.handle_get_grasp_points)

    def handle_get_grasp_points(self,req):
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
        self.data_collector.save_point_cloud_current(save_data)

        time.sleep(0.3)

        # load image, point cloud, distribution
        data = Data()
        data.name = save_data.name
        img_list, mask_list = None, None
        while img_list is None and mask_list is None:
            img_list, mask_list = self.data_monster.load_img_mask([data], self.path + 'current/')
            time.sleep(0.1)

        # generate grasp points
        filter_xyz_dict = self.data_monster.get_all_filter_xyz(data, self.distribution, img_list[0], mask_list[0])

        distribution_cf = self.data_monster.get_distribution_cameraframe(self.distribution, filter_xyz_dict)
        self.data_monster.show_point_cloud(data.name)
        avg_dic = self.data_monster.model_distribution(distribution_cf)

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
    rospy.spin()
    # grasp_generator.handle_get_grasp_points(None)
