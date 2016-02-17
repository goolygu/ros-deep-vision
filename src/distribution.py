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

    def set(self, layer_name, filter_idx, frame_name, point_array):
        filter_idx = int(filter_idx)
        if not layer_name in self.data_dict:
            self.data_dict[layer_name] = {}
        if not filter_idx in self.data_dict[layer_name]:
            self.data_dict[layer_name][filter_idx] = {}

        self.data_dict[layer_name][filter_idx][frame_name] = point_array.tolist()

    def save(self, path, name):
        with open(path + "/distribution/" + name + '.yaml', 'w') as outfile:
            outfile.write( yaml.dump(self) )

    def load(self, path, name):
        f = open(path + "/distribution/" + name + '.yaml')
        dist = yaml.load(f)
        self.data_dict = dist.data_dict
