
import numpy as np
import yaml

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
