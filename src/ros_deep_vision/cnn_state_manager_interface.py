#! /usr/bin/env python
import roslib
import rospy
import sys
from data_monster import *
from data_collector import *
import settings
from perception_msgs.srv import GetListState
from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose
from data_util import *
from data_settings import *
from distribution import *
import time


class CNNStateManagerInterface:
    def __init__(self, settings):
        self.cnn_state_manager = CNNStateManager(settings)
        s = rospy.Service('get_cnn_list_state', GetListState, self.handle_get_cnn_list_state)


    def handle_get_cnn_list_state(self,req):

        if req.state_list[0].name == "None":
            state_list, pose_list = self.cnn_state_manager.get_cnn_list_state(None)
        else:
            state_list, pose_list = self.cnn_state_manager.get_cnn_list_state(req.state_list)

        resp = GetListStateResponse()

        resp.state_list = tuple(state_list)
        resp.pose_list = tuple(pose_list)
        print "send msg"
        return resp

if __name__ == '__main__':
    cnn_state_manager_i = CNNStateManagerInterface(settings)

    rospy.spin()
