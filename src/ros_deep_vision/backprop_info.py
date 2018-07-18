#! /usr/bin/env python


class BackPropInfo:

    def __init__(self, data_settings):
        self.layer_name_list = ['conv5', 'conv4', 'conv3', 'conv2', 'conv1']
        self.num_filters = {}
        self.frame_list = {}
        ds = data_settings

        # training pick top n consistent filters
        self.num_filters['conv5'] = ds.conv5_top
        self.num_filters['conv4'] = ds.conv4_top
        self.num_filters['conv3'] = ds.conv3_top
        self.frame_list['conv5'] = ds.frame_list_conv5
        self.frame_list['conv4'] = ds.frame_list_conv4
        self.frame_list['conv3'] = ds.frame_list_conv3

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
