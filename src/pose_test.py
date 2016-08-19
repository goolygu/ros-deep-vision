#! /usr/bin/env python
import os
import re
from data_monster import *
from data_collector import Data
import settings
import cv2
import pcl
from image_misc import *
import yaml
import numpy as np
from umass_atg.classes.types import *
from umass_atg.pose_state_manager import *
from data_util import *

class CNNState:
    def __init__(self):
        self.xyz_dict = {}
        self.response_dict = {}
        self.angle = 0
        self.name = ""

    def save(self, path, name):
        with open(path + name + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)

    def merge(self, other_state):
        self.xyz_dict.update(other_state.xyz_dict)
        self.response_dict.update(other_state.response_dict)
class Result:

    def __init__(self):
        pass
    def save(self, path, name):
        with open(path + name + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)

class PoseTest:

    def __init__(self, data_path, state_path, ds, case):

        self.mode = "nomerge"#"merge"#

        if case == 2 or self.mode == 'merge':
            self.data_monster = DataMonster(settings, ds)
            self.data_monster.visualize = True
            self.data_monster.show_backprop = False
        self.input_dims = (227,227)
        self.pose_state_manager = PoseStateManager(ds)
        self.data_path = data_path
        self.state_path = state_path
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.ds = ds
    def load_cnn_state(self, name):
        try:
            f = open(self.state_path + name + '.yaml')
            cnn_state = yaml.load(f)
            return cnn_state
        except:
            return None

    def state_to_aspect(self, cnn_state):
        state_list, pose_list = to_state_pose_list(cnn_state.response_dict, cnn_state.xyz_dict)
        aspect = Aspect()
        aspect.set_state_list(state_list)
        pose_dic = Dic2()
        for i, state in enumerate(state_list):
            pose_dic.add(state.type, state.name, pose_list[i])
        pose_state_list = self.pose_state_manager.get_pose_state(pose_dic)
        aspect.set_pose_state_list(pose_state_list)
        return aspect

    def get_crop_pointcloud(self, file_prefix):

        center = [320,240]
        constant = 570.3
        MM_PER_M = 1000

        depth_name = file_prefix + "_depthcrop.png"
        depth = cv2.imread(depth_name, -1).astype("float")

        if self.ds.square == 'inc':
            depth[depth == 0] = np.nan

        # print depth
        top_left_name = file_prefix + "_loc.txt"
        f = open(top_left_name, 'r')
        topleft = f.read().replace("\n","").split(",")
        topleft = [int(num_str) for num_str in topleft]
        # print topleft
        pc = np.zeros(depth.shape + (3,) )

        grid = np.mgrid[1:depth.shape[0]+1, 1:depth.shape[1]+1]

        xgrid = grid[1] +topleft[0]-1 - center[0]
        ygrid = grid[0] +topleft[1]-1 - center[1]
        # print xgrid
        # print ygrid

        pc[:,:,0] = xgrid * depth / constant / MM_PER_M
        pc[:,:,1] = ygrid * depth / constant / MM_PER_M
        pc[:,:,2] = depth / MM_PER_M

        return pc
        # print pc


    def get_data(self, file_prefix):
        print file_prefix
        img_name = file_prefix + "_crop.png"
        img = cv2_read_file_rgb(img_name)
        if self.ds.square == 'dec':
            img = crop_to_square(img)
        else:
            img = increase_to_square(img,0.)
        img = cv2.resize(img, self.input_dims)

        mask_name = file_prefix + "_maskcrop.png"
        mask = cv2.imread(mask_name)
        if self.ds.square == 'dec':
            mask = crop_to_square(mask)
        else:
            mask = increase_to_square(mask,0.)
        mask = np.reshape(mask[:,:,0], (mask.shape[0], mask.shape[1]))
        if self.ds.square == 'inc':
            mask = cv2.dilate(mask, self.dilate_kernel)
        mask = cv2.resize(mask, self.input_dims)

        pc = self.get_crop_pointcloud(file_prefix)
        if self.ds.square == 'dec':
            pc = crop_to_square(pc)
        else:
            pc = increase_to_square(pc,np.nan)

        pose_name = file_prefix + "_pose.txt"
        f = open(pose_name, 'r')
        angle = float(f.read().replace("\n",""))

        data = Data()
        data.img = img
        data.mask = mask
        data.pc = pc
        data.angle = angle

        return data

    def build_states(self):

        cat_list = os.listdir(self.data_path)[42:45]
        for cat_name in cat_list:#["food_bag"]:#os.listdir(self.data_path):#["camera"]:#["apple"]:#
            for ins_name in os.listdir(self.data_path+cat_name):#["camera_3"]:#["apple_1"]:#
                ins_num = re.match(cat_name + "_" + "(.*)", ins_name).group(1)
                for file_name in os.listdir(self.data_path+cat_name+"/"+ins_name):
                    m = re.match(ins_name + "_" + '(.*)_pose\.txt$', file_name)
                    if m:
                        video_frame = m.group(1)
                        video, frame = video_frame.split("_")
                        print cat_name, ins_num, video, frame
                        file_prefix = self.data_path + cat_name + "/" + ins_name + "/" + ins_name + "_" + video_frame

                        # check if state already created
                        if os.path.isfile(self.state_path + ins_name + "_" + video_frame + '.yaml'):
                            continue

                        data = self.get_data(file_prefix)
                        cnn_state = CNNState()
                        cnn_state.xyz_dict, cnn_state.response_dict = self.data_monster.get_state(None, data)

                        cnn_state.angle = data.angle
                        cnn_state.category = cat_name
                        cnn_state.ins_num = ins_num
                        cnn_state.video = video
                        cnn_state.frame = frame

                        cnn_state.save(self.state_path, ins_name + "_" + video_frame)

                        # break

    def angle_diff(self, a1, a2):
        diff = a1 - a2
        diff = (diff + 180) % 360 -180
        return abs(diff)

    def get_load_aspect(self, test_file):
        cnn_state = self.load_cnn_state(test_file)
        if cnn_state == None:
            return None
        aspect = self.state_to_aspect(cnn_state)
        return aspect, cnn_state.angle

    def get_file_prefix_from_file_name(self, file_name):
        # cat_name, ins_num, video, frame
        name_split_list = file_name.split("_")
        frame = name_split_list[-1]
        video = name_split_list[-2]
        ins_num = name_split_list[-3]
        cat_name = "_".join(name_split_list[0:-3])
        ins_name = cat_name + "_" + ins_num

        prefix = self.data_path + cat_name + "/" + ins_name + "/" + ins_name + "_" + video + "_" + frame
        # print prefix
        return prefix

    def get_compare_dist(self, compare_file_list):
        merged_cnn_state = CNNState()
        for compare_file in compare_file_list:
            compare_cnn_state = self.load_cnn_state(compare_file)
            if compare_cnn_state == None:
                continue
            merged_cnn_state.merge(compare_cnn_state)

        merged_aspect = self.state_to_aspect(merged_cnn_state)
        dist = state_list_to_dist(merged_aspect.get_state_list())
        return dist

    def get_obs_aspect(self, dist, test_file):

        file_prefix = self.get_file_prefix_from_file_name(test_file)
        data = self.get_data(file_prefix)
        cnn_state = CNNState()
        cnn_state.xyz_dict, cnn_state.response_dict = self.data_monster.get_state(dist, data)

        aspect = self.state_to_aspect(cnn_state)
        return aspect, data.angle

    def get_test_train_dic(self):
        test_dic = {}
        train_dic = {}

        # build up list of file names for training
        for cat_name in ["food_bag"]:#os.listdir(self.data_path):#["calculator","camera","food_bag","lightbulb","notebook","soda_can"]:#["camera"]:#["apple"]:#
            for ins_name in os.listdir(self.data_path+cat_name):#["camera_3"]:#["apple_1"]:#
                ins_num = re.match(cat_name + "_" + "(.*)", ins_name).group(1)
                for file_name in os.listdir(self.data_path+cat_name+"/"+ins_name):
                    m = re.match(ins_name + "_" + '(.*)_pose\.txt$', file_name)
                    if m:
                        video_frame = m.group(1)
                        video, frame = video_frame.split("_")
                        file_name = ins_name + "_" + video_frame
                        # if test case
                        if video == '2':
                            if not ins_name in test_dic:
                                test_dic[ins_name] = []
                            test_dic[ins_name].append(file_name)
                        else:
                            if not ins_name in train_dic:
                                train_dic[ins_name] = []
                            train_dic[ins_name].append(file_name)

        return test_dic, train_dic

    def test(self):

        test_dic, train_dic = self.get_test_train_dic()

        err_dic = {}
        avg_dic = {}
        med_dic = {}

        err_list = []

        result = Result()



        print "start testing"
        # test
        for ins_name in test_dic: #["notebook_1"]:#
            print ins_name
            err_dic[ins_name] = []

            if self.mode == "merge":
                dist = self.get_compare_dist(test_dic[ins_name])

            for test_file in test_dic[ins_name]:
                print test_file

                if self.mode == "merge":
                    obs_aspect, gt_angle = self.get_obs_aspect(dist, test_file)
                else:
                    obs_aspect, gt_angle = self.get_load_aspect(test_file)


                if obs_aspect == None:
                    continue

                sim_dic = {}
                max_sim = 0
                best_angle = 0

                for compare_file in train_dic[ins_name]:
                    compare_cnn_state = self.load_cnn_state(compare_file)
                    if compare_cnn_state == None:
                        continue

                    aspect = self.state_to_aspect(compare_cnn_state)

                    sim = aspect.observation_similarity(obs_aspect)
                    sim_dic[compare_cnn_state.angle] = sim

                    if sim > max_sim:
                        max_sim = sim
                        best_angle = compare_cnn_state.angle

                err = self.angle_diff(best_angle, gt_angle)
                err_dic[ins_name].append(err)
                err_list.append(err)
                print err

            if len(err_dic[ins_name]) == 0:
                continue

            avg = sum(err_dic[ins_name])/float(len(err_dic[ins_name]))
            avg_dic[ins_name] = avg
            med = np.median(np.array(err_dic[ins_name]))
            med_dic[ins_name] = float(med)

            print "\ninstance", ins_name, "avg", avg, "med", med, "\n"

        if len(err_list) == 0:
            return
        avg = sum(err_list)/float(len(err_list))
        med = np.median(np.array(err_list))

        print "avg", avg
        print "med", med

        result.med_dic = med_dic
        result.avg_dic = avg_dic
        result.avg = avg
        result.med = float(med)

        result.save(self.state_path, "result_" + self.mode)

if __name__ == '__main__':

    ds = DataSettings()
    case = 2
    name = ds.get_pose_state_name()

    state_path = "/home/lku/Workspace/WRGBD_Test/" + name + "/"

    if not os.path.exists(state_path):
        os.makedirs(state_path)

    pose_test = PoseTest("/home/lku/Dataset/rgbd-dataset/",state_path, ds, case)
    if case == 1:
        pose_test.test()
    elif case == 2:
        pose_test.build_states()
    pass
