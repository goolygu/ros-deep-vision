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
from umass_atg.pose_state_manager import *
import tf
import rospkg

class CNNConditionManagerInterface:
    def __init__(self, settings):

        replay_dir = None

        if rospy.has_param("~replay_data"):
            replay_data = rospy.get_param("~replay_data")
            rospy.delete_param("~replay_data")
            print "replay_data: ", replay_data

            if replay_data == "no_replay":
                replay_dir = None
                self.replay_mode = False
            else:
                rospack = rospkg.RosPack()
                replay_dir = rospack.get_path(replay_data) + "/current/"
                self.replay_mode = True

        self.cnn_state_manager = CNNStateManager(settings, data_setting_case="r2_demo", replay_dir=replay_dir)
        # self.cnn_state_manager.set_box_param(200, 0, 15)
        self.cnn_state_manager.set_box_param(185, 0, 15, 185)
        # self.cnn_state_manager.set_box_param(150, 0, 15, 150)

        s = rospy.Service('get_cnn_condition', GetCondition, self.handle_get_cnn_condition)

        self.called = False
        self.req = None

        self.pose_state_manager = PoseStateManager(ds=self.cnn_state_manager.ds)
        self.tf_listener = tf.TransformListener()
        self.pose_state_manager.set_tf_listener(self.tf_listener)

    # list of Dic2 of value, list of Dic2 of xyz, list of vector
    def create_condition(self, value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name):
        condition = Condition()
        condition.name = name
        # loop through aspects
        for i, value_dic in enumerate(value_dic_list):

            xyz_dic = xyz_dic_list[i]
            xy_dic = xy_dic_list[i]
            aspect = Aspect()
            aspect_pose = AspectPose()

            # for aspect
            state_list = []
            pose_dic = Dic2()
            # for aspect_pose
            state_type_list = []
            state_name_list = []
            pose_list = []
            xy_list = []

            # loop through each feature
            for sig in value_dic:
                state = State()
                state.type = 'cnn'
                state.name = str(sig)
                state.value = value_dic[sig]
                state_list.append(state)

                state_type_list.append(state.type)
                state_name_list.append(state.name)

                pose_msg = Pose()
                xy_msg = Point()
                if not state.value == 0:
                    pose_msg.position.x = xyz_dic[sig][0]
                    pose_msg.position.y = xyz_dic[sig][1]
                    pose_msg.position.z = xyz_dic[sig][2]
                    pose_msg.orientation.x = 0
                    pose_msg.orientation.y = 0
                    pose_msg.orientation.z = 0
                    pose_msg.orientation.w = 1
                    xy_msg.x = xy_dic[sig][0]
                    xy_msg.y = xy_dic[sig][1]
                    xy_msg.z = 0.
                pose_list.append(pose_msg)
                xy_list.append(xy_msg)
                pose_dic.add(state.type, state.name, pose_msg)

            pose_state_list = self.pose_state_manager.get_pose_state(pose_dic)
            aspect.set_state_list(state_list)
            aspect.set_pose_state_list(pose_state_list)
            aspect.set_img_src(img_name_list[i].split("/")[-1])

            aspect_pose.set(state_type_list, state_name_list, pose_list, xy_list)

            condition.aspect_name_list.append(aspect.img_src)
            condition.aspect_list.append(aspect)
            condition.aspect_pose_list.append(aspect_pose)

        for centroid in centroid_list:
            point = Point()
            point.x = centroid[0]
            point.y = centroid[1]
            point.z = centroid[2]
            condition.aspect_loc_list.append(point)

        return condition

    def get_clustered_cnn_list_state(self):

        req = self.req
        if req.expected_condition.name == "None":
            value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name = self.cnn_state_manager.get_clustered_cnn_list_state(None,None)
        elif req.expected_condition.name == "":
            value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name = \
            self.cnn_state_manager.get_clustered_cnn_list_state(req.expected_condition.aspect_list, req.aspect_idx_list,use_last_observation=True, data_name=req.expected_condition.name)
        else:
            value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name = \
            self.cnn_state_manager.get_clustered_cnn_list_state(None, None, data_name=req.expected_condition.name)

        resp = GetConditionResponse()

        # condition = Condition()
        # condition.set(value_dic_list, xyz_dic_list, centroid_list, img_name_list)
        condition = self.create_condition(value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name)
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
    rospy.init_node('cnn_condition_manager_interface', anonymous=False)
    cnn_condition_manager_i = CNNConditionManagerInterface(settings)
    cnn_condition_manager_i.loop()
    # rospy.spin()
