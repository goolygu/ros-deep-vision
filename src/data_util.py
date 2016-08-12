
from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose
from distribution import *

def state_list_to_dist(state_list):
    dist = Distribution()
    for state in state_list:
        # print "state", state.name
        dist.set_tree_feature(state.name)
    return dist

def to_state_pose_list(value_dict, filter_xyz_dict):
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
    return state_list, pose_list

class Dic2:

    def __init__(self):
        self.dic = {}

    def __repr__(self):
        return repr(self.dic)

    def add(self,k1,k2,value):
        if not k1 in self.dic:
            self.dic[k1] = {}

        self.dic[k1][k2] = value

    def has(self,k1,k2):
        if k1 in self.dic:
            if k2 in self.dic[k1]:
                return True

        return False

    def get(self,k1,k2):

        return self.dic[k1][k2]

    def set(self,k1,k2,value):
        if self.has(k1,k2):
            self.dic[k1][k2] = value
            return True
        else:
            return False

    def keys1(self):
        return self.dic.keys()

    def keys2(self, k1):
        return self.dic[k1].keys()

    def get_sublist(self, k1):
        sublist = []
        for k2 in self.dic[k1]:
            value = self.dic[k1][k2]
            sublist.append((k2,value))
        return sublist

    def get_list(self):
        tuple_list = []
        for k1 in self.dic:
            for k2 in self.dic[k1]:
                value = self.dic[k1][k2]
                tuple_list.append((k1,k2,value))
        return tuple_list

    def combine(self, dic_other):
        for k1 in dic_other.dic:
            for k2 in dic_other.dic[k1]:
                value = dic_other.dic[k1][k2]
                self.add(k1,k2,value)

    def clear(self):
        self.dic.clear()
