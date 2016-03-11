#! /usr/bin/env python
# -*- coding: utf-8
import sys
import os
import numpy as np
import time

import yaml
from sklearn.neighbors.kde import KernelDensity
import tf

class Distribution:
    def __init__(self):
        # data dict stores the points relative to a certain parent filters and filter.
        self.data_dict = {}
        # filter tree stores a hierarchical structure of filter ids
        self.filter_tree = {}

    def merge(self, dist):
        for sig in dist.data_dict:
            if sig in self.data_dict:
                self.data_dict[sig] = self.data_dict[sig] + dist.data_dict[sig]
            else:
                self.data_dict[sig] = dist.data_dict[sig]

        self.rec_merge(self.data_dict, dist)


    def rec_merge(self, d1, d2):
        '''update first dict with second recursively'''
        for k, v in d1.iteritems():
            if k in d2:
                d2[k] = rec_merge2(v, d2[k])
        d1.update(d2)
        return d1

    # saves a list of filters under the same parents
    def set_tree_list(self, parent_filters, new_filter_list):
        tree = self.filter_tree
        for i, parent in enumerate(parent_filters):
            tree = tree[int(parent)]
        for new_filter in new_filter_list:
            tree[int(new_filter)] = {}

    # save a single filter
    def set_tree(self, parent_filters, new_filter):
        tree = self.filter_tree
        for i, parent in enumerate(parent_filters):
            tree = tree[int(parent)]

        tree[int(new_filter)] = {}

    # filter_sig is a list of filter ids from parent to current
    def set(self, filter_sig, frame_name, point_array):
        filter_sig = [int(id) for id in filter_sig]
        filter_sig = tuple(filter_sig)
        if not filter_sig in self.data_dict:
            self.data_dict[filter_sig] = {}

        if type(point_array) == type(np.array([])):
            self.data_dict[filter_sig][frame_name] = point_array.tolist()
        else:
            self.data_dict[filter_sig][frame_name] = point_array
    # def set(self, layer_name, filter_idx, frame_name, point_array):
    #     filter_idx = int(filter_idx)
    #     if not layer_name in self.data_dict:
    #         self.data_dict[layer_name] = {}
    #     if not filter_idx in self.data_dict[layer_name]:
    #         self.data_dict[layer_name][filter_idx] = {}
    #
    #     self.data_dict[layer_name][filter_idx][frame_name] = point_array.tolist()

    def save_exact(self, path, name):
        with open(path + name + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)

    def save(self, path, name):
        with open(path + "/distribution/" + name + '.yaml', 'w') as f:
            yaml.dump(self, f, default_flow_style=False)
        # with open(path + "/distribution/" + name + '.yaml', 'w') as outfile:
        #     outfile.write( yaml.dump(self) )

    def load(self, path, name):
        f = open(path + "/distribution/" + name + '.yaml')
        dist = yaml.load(f)
        self.data_dict = dist.data_dict
        self.filter_tree = dist.filter_tree

def remove_nan(point_list):
    new_list = np.array([]).reshape([0,3])
    for point in point_list:
        if not np.any(np.isnan(point)):
            new_list = np.append(new_list, [point], axis=0)
    return new_list

def find_max_density(point_list):
    point_list = remove_nan(point_list)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(point_list)
    points = kde.sample(100000)
    prob_list = kde.score_samples(points)
    max_point = points[np.argmax(prob_list)]
    print "max", max_point
    return max_point
