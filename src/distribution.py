#! /usr/bin/env python
# -*- coding: utf-8
import sys
import os
import numpy as np
import time

import yaml

import tf

class Distribution:
    def __init__(self):
        self.data_dict = {}
        self.filter_tree = {}

    def set_tree_list(self, parent_filters, new_filter_list):
        tree = self.filter_tree
        for i, parent in enumerate(parent_filters):
            tree = tree[int(parent)]
        for new_filter in new_filter_list:
            tree[int(new_filter)] = {}

    def set_tree(self, parent_filters, new_filter):
        tree = self.filter_tree
        for i, parent in enumerate(parent_filters):
            tree = tree[int(parent)]

        tree[int(new_filter)] = {}

    def set(self, filter_sig, frame_name, point_array):
        filter_sig = [int(id) for id in filter_sig]
        filter_sig = tuple(filter_sig)
        if not filter_sig in self.data_dict:
            self.data_dict[filter_sig] = {}

        self.data_dict[filter_sig][frame_name] = point_array.tolist()

    # def set(self, layer_name, filter_idx, frame_name, point_array):
    #     filter_idx = int(filter_idx)
    #     if not layer_name in self.data_dict:
    #         self.data_dict[layer_name] = {}
    #     if not filter_idx in self.data_dict[layer_name]:
    #         self.data_dict[layer_name][filter_idx] = {}
    #
    #     self.data_dict[layer_name][filter_idx][frame_name] = point_array.tolist()

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
