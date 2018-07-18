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
from umass_atg.classes.types import *
from umass_atg.pose_state_manager import *
import tf
import rospkg

class CNNConditionManagerInterface:
    def __init__(self, settings):

        replay_pkg = "ros_deep_vision"

        if rospy.has_param("~replay_pkg"):
            replay_pkg = rospy.get_param("~replay_pkg")
            rospy.delete_param("~replay_pkg")

        print "replay_pkg: ", replay_pkg

        replay_folder = "current"
        if rospy.has_param("~replay_folder"):
            replay_folder = rospy.get_param("~replay_folder")
            rospy.delete_param("~replay_folder")

        print "replay_folder: ", replay_folder

        rospack = rospkg.RosPack()
        replay_dir = rospack.get_path(replay_pkg)# + "/" + replay_folder + "/"

        self.cnn_state_manager = CNNStateManager(settings, data_setting_case="r2_ratchet_demo", replay_dir=replay_dir, replay_folder=replay_folder)
        # self.cnn_state_manager.set_box_param(200, 0, 15)
        # self.cnn_state_manager.set_box_param(185, 0, 15, 185)
        self.cnn_state_manager.set_box_param(200, 0, 15, 200, left_hand_offset=True)

        s = rospy.Service('get_cnn_condition', GetCondition, self.handle_get_cnn_condition)

        self.called = False
        self.req = None

        self.pose_state_manager = PoseStateManager(ds=self.cnn_state_manager.ds)
        self.tf_listener = tf.TransformListener()
        self.pose_state_manager.set_tf_listener(self.tf_listener)

    def create_aspect(self, value_dic, xyz_dic, xy_dic, img_name):

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
        aspect.set_img_src(img_name)

        aspect_pose.set(state_type_list, state_name_list, pose_list, xy_list)

        return aspect, aspect_pose

    # list of Dic2 of value, list of Dic2 of xyz, list of vector
    def create_condition(self, cnn_list_state):#value_dic_list, xyz_dic_list, xy_dic_list, centroid_list, img_name_list, name):
        condition = Condition()
        condition.name = cnn_list_state.name
        # loop through aspects
        for i, value_dic in enumerate(cnn_list_state.value_dic_list):

            xyz_dic = cnn_list_state.xyz_dic_list[i]
            xy_dic = cnn_list_state.xy_dic_list[i]
            img_name = cnn_list_state.img_name_list[i].split("/")[-1]

            aspect, aspect_pose = self.create_aspect(value_dic, xyz_dic, xy_dic, img_name)

            condition.aspect_name_list.append(aspect.img_src)
            condition.aspect_list.append(aspect)
            condition.aspect_pose_list.append(aspect_pose)

        for centroid in cnn_list_state.centroid_list:
            point = Point()
            point.x = centroid[0]
            point.y = centroid[1]
            point.z = centroid[2]
            condition.aspect_loc_list.append(point)

        return condition

    def get_cnn_condition(self, req):
        print "got request, mode", req.mode

        self.cnn_state_manager.set_mode(req.mode)
        self.pose_state_manager.set_mode(req.mode)
        cnn_req = CNNStateRequest()

        if req.mode == "replay":
            print "replay"
            cnn_req.data_name = req.expected_condition.name
            cnn_list_state = self.cnn_state_manager.get_clustered_cnn_list_state(cnn_req)

        # no expectation on what will observe
        elif req.expected_condition.name == "None":
            print "default"
            cnn_list_state = self.cnn_state_manager.get_clustered_cnn_list_state(cnn_req)
        # refining
        elif req.expected_condition.name == "" and len(req.aspect_idx_list) > 0:
            print "refine"
            cnn_req.expected_aspect_list = req.expected_condition.aspect_list
            cnn_req.aspect_idx_list = req.aspect_idx_list
            cnn_req.use_last_observation = True
            cnn_req.data_name = req.expected_condition.name
            cnn_req.mode = "refine"
            cnn_list_state = self.cnn_state_manager.get_clustered_cnn_list_state(cnn_req)

        # refine action
        elif not req.expected_condition is None:
            print "refine action"
            cnn_req.expected_aspect_list = req.expected_condition.aspect_list
            cnn_req.use_last_observation = False
            cnn_req.mode = "refine_action"
            cnn_list_state = self.cnn_state_manager.get_clustered_cnn_list_state(cnn_req)
        # replay
        else:
            print "unknow situation"

        # resp = GetConditionResponse()

        # condition = Condition()
        # condition.set(value_dic_list, xyz_dic_list, centroid_list, img_name_list)
        condition = self.create_condition(cnn_list_state)

        # only keep most similar aspect
        if cnn_req.mode == "refine_action":
            actor_aspect = Aspect()
            actor_aspect.from_ros_msg(cnn_req.expected_aspect_list[0])
            max_prob = -sys.float_info.max
            max_idx = None
            for i, aspect in enumerate(condition.aspect_list):
                sim_log_prob = actor_aspect.log_appearance_similarity(aspect)
                if max_prob < sim_log_prob:
                    max_prob = sim_log_prob
                    max_idx = i
            condition.keep_idx(max_idx)


        return condition
        # resp.condition = condition.to_ros_msg()
        # self.resp = resp

    # a hack, for some reason cnn is slow in ros handler
    def loop(self):

        while not rospy.is_shutdown():
            if self.called == True:
                condition = self.get_cnn_condition(self.req)
                resp = GetConditionResponse()
                resp.condition = condition.to_ros_msg()
                self.resp = resp
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
