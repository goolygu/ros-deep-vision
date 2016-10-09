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
        self.err_dic = {}
        self.avg_dic = {}
        self.med_dic = {}
        pass
    def save(self, path, name):
        with open(path + name + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)



class PoseTest:

    def __init__(self, data_path, state_path, ds, case):

        self.mode = "nomerge"#"merge"#

        if case == 2 or case == 0 or self.mode == 'merge':
            self.data_monster = DataMonster(settings, ds)
            self.data_monster.visualize = True
            self.data_monster.show_backprop = False
        self.input_dims = (227,227)
        self.pose_state_manager = PoseStateManager(ds)
        self.data_path = data_path
        self.state_path = state_path
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.ds = ds

    def load_result(self, path, name):
        try:
            f = open(path + name + '.yaml')
            result = yaml.load(f)
            return result
        except:
            print "failed loading", name
            return None

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

    def fill_depth(self, depth):

        show = False
        if show:
            s_depth = np.round(depth/float(np.amax(depth))*255.0).astype("uint8")
            print s_depth
            cv2.imshow("depth", s_depth)
            cv2.waitKey(100)
        new_depth = np.copy(depth)
        for x in range(depth.shape[0]):
            for y in range(depth.shape[1]):
                if depth[x,y] == 0:
                    new_depth[x,y] = closest_value_fast(depth,x,y)
        if show:
            s_new_depth = np.round(new_depth/float(np.amax(new_depth))*255.0).astype("uint8")
            print s_new_depth
            cv2.imshow("depth inpaint", s_new_depth)
            cv2.waitKey(100)
        return new_depth

    def build_filled_depth_imgs(self):
        for cat_name in os.listdir(self.data_path)[::1]:
            for ins_name in os.listdir(self.data_path+cat_name)[::1]:
                ins_num = re.match(cat_name + "_" + "(.*)", ins_name).group(1)
                for file_name in os.listdir(self.data_path+cat_name+"/"+ins_name)[::1]:
                    m = re.match(ins_name + "_" + '(.*)_pose\.txt$', file_name)
                    if m:
                        video_frame = m.group(1)
                        video, frame = video_frame.split("_")
                        print cat_name, ins_num, video, frame

                        file_prefix = self.data_path + cat_name + "/" + ins_name + "/" + ins_name + "_" + video_frame
                        depth_name = file_prefix + "_depthcrop.png"

                        # skip if already created
                        if os.path.isfile(file_prefix + "_filldepthcrop.npy"):
                            continue

                        depth = cv2.imread(depth_name, -1).astype("float")
                        filled_depth = self.fill_depth(depth)
                        np.save(file_prefix + "_filldepthcrop", filled_depth)

                        # cmp_filled_depth = np.load(file_prefix + "_filldepthcrop.npy")
                        # print cmp_filled_depth
                        # print filled_depth
                        # raw_input()
                        # return

    def get_crop_pointcloud(self, file_prefix):

        center = [320,240]
        constant = 570.3
        MM_PER_M = 1000

        if self.ds.cloud_gap == 'inpaint':
            # depth = self.fill_depth(depth)
            depth_name = file_prefix + "filldepthcrop.npy"
            depth = np.load(file_prefix + "_filldepthcrop.npy")
            if depth == None:
                print "no depth"
                return
        else:
            depth_name = file_prefix + "_depthcrop.png"
            depth = cv2.imread(depth_name, -1).astype("float")
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

        cat_list = os.listdir(self.data_path)
        for cat_name in cat_list[::-1]:#["food_bag"]:#os.listdir(self.data_path):#["camera"]:#["apple"]:#
            for ins_name in os.listdir(self.data_path+cat_name)[::-1]:#["camera_3"]:#["apple_1"]:#
                ins_num = re.match(cat_name + "_" + "(.*)", ins_name).group(1)
                for file_name in os.listdir(self.data_path+cat_name+"/"+ins_name)[::-1]:
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
            return None, None
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
        for cat_name in os.listdir(self.data_path):#["food_bag"]:#["calculator","camera","food_bag","lightbulb","notebook","soda_can"]:#["camera"]:#["apple"]:#
            test_dic[cat_name] = {}
            train_dic[cat_name] = {}
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
                            if not ins_name in test_dic[cat_name]:
                                test_dic[cat_name][ins_name] = []
                            test_dic[cat_name][ins_name].append(file_name)
                        else:
                            if not ins_name in train_dic[cat_name]:
                                train_dic[cat_name][ins_name] = []
                            train_dic[cat_name][ins_name].append(file_name)

        return test_dic, train_dic

    def test(self):

        test_dic, train_dic = self.get_test_train_dic()

        result = Result()

        err_list = []
        # print test_dic.keys()
        # return
        print "start testing"
        # test
        for cat_name in test_dic.keys()[0::1]:
            print cat_name
            result_cat = Result()
            # result_file_cat_name = "result_" + self.mode + "_" + cat_name
            result_file_cat_name = "result_" + self.ds.get_pose_state_test_name() + "_" + self.mode + "_" + cat_name
            if os.path.isfile(self.state_path + result_file_cat_name + '.yaml'):
                result_cat = self.load_result(self.state_path, result_file_cat_name)
                for ins_name in test_dic[cat_name]:
                    err_list += result_cat.err_dic[ins_name]
                    result.err_dic[ins_name] = result_cat.err_dic[ins_name]
                    result.avg_dic[ins_name] = result_cat.avg_dic[ins_name]
                    result.med_dic[ins_name] = result_cat.med_dic[ins_name]
                continue



            err_list_cat = []
            for ins_name in test_dic[cat_name].keys(): #["notebook_1"]:#
                print ins_name
                result.err_dic[ins_name] = []
                result_cat.err_dic[ins_name] = []

                # create compare list of aspect
                cmp_list = []
                for compare_file in train_dic[cat_name][ins_name]:
                    cmp_aspect, cmp_angle = self.get_load_aspect(compare_file)
                    cmp_list.append((cmp_aspect,cmp_angle))

                if self.mode == "merge":
                    dist = self.get_compare_dist(test_dic[cat_name][ins_name])

                for test_file in test_dic[cat_name][ins_name]:
                    print test_file

                    if self.mode == "merge":
                        obs_aspect, gt_angle = self.get_obs_aspect(dist, test_file)
                    else:
                        obs_aspect, gt_angle = self.get_load_aspect(test_file)

                    if obs_aspect == None:
                        continue

                    # sim_dic = {}
                    max_sim = -float("inf")
                    best_angle = 0

                    for cmp_tuple in cmp_list:
                        if self.ds.similarity == "L1":
                            sim  = cmp_tuple[0].L1_similarity(obs_aspect)
                        elif self.ds.similarity == "custom":
                            sim  = cmp_tuple[0].observation_similarity(obs_aspect)
                        elif self.ds.similarity == "clean":
                            sim = cmp_tuple[0].similarity(obs_aspect)
                        elif self.ds.similarity == "gaussian":
                            sim = cmp_tuple[0].gaussian_similarity(obs_aspect)
                        elif self.ds.similarity == "heavytail":
                            sim = cmp_tuple[0].ht_similarity(obs_aspect)
                        if sim > max_sim:
                            max_sim = sim
                            best_angle = cmp_tuple[1]

                    # for compare_file in train_dic[cat_name][ins_name]:
                    #
                    #     cmp_aspect, cmp_angle = self.get_load_aspect(compare_file)
                    #     # if cmp_aspect == None:
                    #     #     continue
                    #     sim = cmp_aspect.observation_similarity(obs_aspect)
                    #     # sim_dic[cmp_angle] = sim
                    #
                    #     if sim > max_sim:
                    #         max_sim = sim
                    #         best_angle = cmp_angle

                    err = self.angle_diff(best_angle, gt_angle)
                    result.err_dic[ins_name].append(err)
                    result_cat.err_dic[ins_name].append(err)
                    err_list.append(err)
                    err_list_cat.append(err)
                    print err
                #
                # if len(result.err_dic[ins_name]) == 0:
                #     continue

                ins_avg = sum(result.err_dic[ins_name])/float(len(result.err_dic[ins_name]))
                result.avg_dic[ins_name] = ins_avg
                result_cat.avg_dic[ins_name] = ins_avg
                ins_med = float(np.median(np.array(result.err_dic[ins_name])))
                result.med_dic[ins_name] = ins_med
                result_cat.med_dic[ins_name] = ins_med

                print "\ninstance", ins_name, "avg", ins_avg, "med", ins_med, "\n"
            print "save temporary"

            result_cat.avg = sum(err_list_cat)/float(len(err_list_cat))
            result_cat.med = float(np.median(np.array(err_list_cat)))
            result_cat.save(self.state_path, result_file_cat_name)

        # if len(err_list) == 0:
        #     return
        result.avg = sum(err_list)/float(len(err_list))
        result.med = float(np.median(np.array(err_list)))

        print "avg", result.avg
        print "med", result.med

        result.save(self.state_path, "result_" + self.ds.get_pose_state_test_name() + "_" + self.mode)

if __name__ == '__main__':

    ds = DataSettings(20)
    case = 0
    name = ds.get_pose_state_name()

    state_path = "/home/lku/Workspace/WRGBD_Test/" + name + "/"

    if not os.path.exists(state_path):
        os.makedirs(state_path)

    pose_test = PoseTest("/home/lku/Dataset/rgbd-dataset/",state_path, ds, case)
    if case == 0:
        pose_test.build_states()
        pose_test.test()
    elif case == 1:
        pose_test.test()
    elif case == 2:
        pose_test.build_states()
    elif case == 3:
        pose_test.build_filled_depth_imgs()
    pass
