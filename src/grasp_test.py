#! /usr/bin/env python
# -*- coding: utf-8
import roslib
import rospy
import sys
import os
import dataset_list as dl
from data_settings import DataSettings
from data_monster import *

import settings

if __name__ == '__main__':
    rospy.init_node('grasp_test', anonymous=True)

    case = "tbp"
    # case = "notbp-test"
    # case = "notbp-train"
    # case = "notbp-conv5"
    # case = "single-conv5"
    ds = DataSettings(case)
    tbp = ds.tbp
    data_monster = DataMonster(settings, ds)

    train_path = "/home/lku/Dataset/r2_grasping_dataset/single/"
    clutter_dataset_path = "/home/lku/Dataset/r2_grasping_dataset/clutter/"

    data_monster.set_train_path(train_path)
    mode = 4

    dist_path = settings.ros_dir + '/distribution/'

    name = ds.get_name()
    # train
    if mode == 0:
        dist_dic = data_monster.train_each_case(tbp, dl.data_name_list)
        for case in dist_dic:
            dist_dic[case].save(dist_path, "[" + case + "]" + name)
    # test single example
    elif mode == 1:
        data_monster.show_backprop = True#False#
        distribution = Distribution()
        case1 = '[side_wrap:cylinder]'
        case2 = '[side_wrap:cuboid]'
        distribution.load(dist_path + '/cross_validation/', case1 + '[leave_yellowjar]' + name)

        distribution = data_monster.filter_distribution(distribution, ds.filter_low_n)

        data_monster.input_manager.set_visualize(True)
        data_list = [data_monster.input_manager.get_data_by_name(train_path,dl.data_name_list[112])]
        diff_avg_dic, diff_dist_dic, diff_fail = data_monster.test_accuracy(distribution, data_list, tbp)
        print diff_dist_dic

    # filter
    elif mode == 2:
        distribution = Distribution()
        case = '[side_wrap:cylinder]'
        distribution.load(dist_path, case + name)
        new_dist = data_monster.filter_distribution(distribution, 3)
        new_dist.save(dist_path, case + name + '_f3')
    # print names
    elif mode == 3:
        name_list = data_monster.input_manager.get_data_name_all(train_path)
        for i, name in enumerate(name_list):
            print "'" + name + "'" , ", # ", i
    # cross validation
    elif mode == 4:

        retrain = True#False#
        data_monster.cross_validation(settings.ros_dir, name, retrain, tbp, dl.data_name_list)

    # merge
    elif mode == 5:
        dist1 = Distribution()
        case1 = '[side_wrap:cylinder]'
        dist1.load(dist_path, case1 + name)

        dist2 = Distribution()
        case2 = '[side_wrap:cuboid]'
        dist2.load(dist_path, case2 + name)

        dist1.merge(dist2)
        dist1.save(dist_path, case1 + case2 + name)

    # test cluttered scenario
    elif mode == 6:
        data_monster.visualize = False#True
        data_monster.show_backprop = False
        case1 = 'side_wrap:cylinder'
        case2 = 'side_wrap:cuboid'

        data_monster.test_clutter(settings.ros_dir, clutter_dataset_path, name, tbp, [case1, case2], None)
    print "done"
    raw_input()
