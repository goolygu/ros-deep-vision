#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from perception_msgs.srv import GetListState, GetListStateResponse
from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose
from data_util import *
from data_settings import *
from distribution import *
import time
from cnn_state_manager import *
from umass_util import *

class CNNStateManagerInterface:
    def __init__(self, settings):

        self.cnn_state_manager = CNNStateManager(settings)

        s = rospy.Service('get_cnn_list_state', GetListState, self.handle_get_cnn_list_state)

        self.called = False
        self.req = None

    def get_cnn_list_state(self):

        req = self.req
        if req.state_list[0].name == "None":
            value_dict, filter_xyz_dict = self.cnn_state_manager.get_cnn_list_state(None)
        else:
            value_dict, filter_xyz_dict = self.cnn_state_manager.get_cnn_list_state(req.state_list)

        state_list, pose_list = to_state_pose_msg_list(value_dict, filter_xyz_dict)
        resp = GetListStateResponse()

        resp.state_list = tuple(state_list)
        resp.pose_list = tuple(pose_list)
        print "send msg"
        self.resp = resp

    # a hack, for some reason cnn is slow in ros handler
    def loop(self):

        while not rospy.is_shutdown():
            if self.called == True:
                self.get_cnn_list_state()
                self.called = False
            rospy.sleep(0.1)

    def handle_get_cnn_list_state(self,req):

        if self.called == False:
            self.called = True
            self.req = req

        while self.called == True:
            rospy.sleep(0.1)
        return self.resp


if __name__ == '__main__':
    rospy.init_node('cnn_state_manager_interface', anonymous=True)
    cnn_state_manager_i = CNNStateManagerInterface(settings)
    cnn_state_manager_i.loop()
    # rospy.spin()
