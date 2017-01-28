#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from perception_msgs.srv import GetCondition, GetConditionResponse
from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose
from data_util import *
from data_settings import *
from distribution import *
import time
from cnn_state_manager import *
from umass_atg.classes.condition import *

class CNNConditionManagerInterface:
    def __init__(self, settings):

        self.cnn_state_manager = CNNStateManager(settings, data_setting_case="r2_demo")
        self.cnn_state_manager.set_box_param(200, 0, 15)
        s = rospy.Service('get_cnn_condition', GetCondition, self.handle_get_cnn_condition)

        self.called = False
        self.req = None


    def get_clustered_cnn_list_state(self):

        req = self.req
        if req.expected_condition.name == "None":
            value_dic_list, xyz_dic_list, centroid_list, img_name_list = self.cnn_state_manager.get_clustered_cnn_list_state(None,None)
        else:
            value_dic_list, xyz_dic_list, centroid_list, img_name_list = \
            self.cnn_state_manager.get_clustered_cnn_list_state(req.expected_condition.aspect_list[0].state_list, req.aspect_idx_list[0])

        resp = GetConditionResponse()

        condition = Condition()
        condition.set(value_dic_list, xyz_dic_list, centroid_list, img_name_list)

        resp.condition = condition.to_ros_msg()
        self.resp = resp

    # a hack, for some reason cnn is slow in ros handler
    def loop(self):

        while not rospy.is_shutdown():
            if self.called == True:
                self.get_clustered_cnn_list_state()
                self.called = False
            rospy.sleep(0.1)

    def handle_get_cnn_condition(self,req):

        if self.called == False:
            self.called = True
            self.req = req

        while self.called == True:
            rospy.sleep(0.1)
        return self.resp


if __name__ == '__main__':
    rospy.init_node('cnn_condition_manager_interface', anonymous=True)
    cnn_condition_manager_i = CNNConditionManagerInterface(settings)
    cnn_condition_manager_i.loop()
    # rospy.spin()
