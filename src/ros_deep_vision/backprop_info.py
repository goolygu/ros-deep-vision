#! /usr/bin/env python


class BackPropInfo:

    def __init__(self, data_settings):
        self.layer_name_list = ['conv5', 'conv4', 'conv3', 'conv2', 'conv1']
        self.num_filters = {}
        self.frame_list = {}
        case = data_settings.case
        # self.top_n_filters = {}
        if case == 'tbp':
            # training pick top n consistent filters
            self.num_filters['conv5'] = 5
            self.num_filters['conv4'] = 5
            self.num_filters['conv3'] = 5
            self.frame_list['conv4'] = ["r2/left_palm"]
            self.frame_list['conv3'] = ["r2/left_thumb_tip","r2/left_index_tip"]
            # testing pick top n filters
            # self.top_n_filters['conv5'] = 5
            # self.top_n_filters['conv4'] = 5
            # self.top_n_filters['conv3'] = 5
        elif case == 'r2_demo':
            self.num_filters['conv5'] = 30
            self.num_filters['conv4'] = 5
            self.num_filters['conv3'] = 0
            # testing pick top n filters
            # self.top_n_filters['conv5'] = 30
            # self.top_n_filters['conv4'] = 5
            # self.top_n_filters['conv3'] = 0
        else:
            raise NotImplementedError("[Error] backprop info no such case" + case)

    def get_layer_name(self):
        return self.layer_name_list[0]

    def get_next_layer_name(self):
        if len(self.layer_name_list) > 1:
            return self.layer_name_list[1]
        else:
            return None

    def get_num_filters(self, layer_name):
        if layer_name in self.num_filters:
            return self.num_filters[layer_name]
        else:
            return 0

    # def get_top_n_filters(self, layer_name):
    #     if layer_name in self.top_n_filters:
    #         return self.top_n_filters[layer_name]
    #     else:
    #         return 0

    def get_frame_list(self, layer_name):
        if layer_name in self.frame_list:
            return self.frame_list[layer_name]
        else:
            return []

    def pop(self):
        self.layer_name_list = self.layer_name_list[1:]
