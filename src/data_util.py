
import numpy as np
import yaml
import re
import os
from data_collector import Data

def get_data_by_name(path, name):

    f = open(path+name+"_data.yaml")
    data = yaml.load(f)
    return data

# returns a dictionary with action type: target type as keys and stores a list of data
def get_data_dic(path, name_list):

    data_dic = {}

    for data_name in name_list:
        data = get_data_by_name(path, data_name)
        key = data.action + ":" + data.target_type
        if key not in data_dic:
            data_dic[key] = []
        data_dic[key].append(data)
        # data_dict[data.name] = data
    return data_dic

# returns a dictionary of dictionary with form dic[action:type][object_name]
def get_data_name_dic(path, name_list):

    data_dic = {}

    for data_name in name_list:
        data = get_data_by_name(path, data_name)
        key = data.action + ":" + data.target_type
        object = data.name.split('_')[0]
        if key not in data_dic:
            data_dic[key] = {}
        if object not in data_dic[key]:
            data_dic[key][object] = []

        data_dic[key][object].append(data)
        # data_dict[data.name] = data
    return data_dic

# returns a list that contains all data
def get_data_all(path):
    data_file_list = []
    match_flags = re.IGNORECASE
    for filename in os.listdir(path):
        if re.match('.*_data\.yaml$', filename, match_flags):
            data_file_list.append(filename)

    data_file_list = sorted(data_file_list)
    print data_file_list
    # data_dict = {}
    data_list = []

    for data_file in data_file_list:
        f = open(path+data_file)
        data = yaml.load(f)
        data_list.append(data)
        # data_dict[data.name] = data
    return data_list

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
