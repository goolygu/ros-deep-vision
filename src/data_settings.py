#! /usr/bin/env python
class DataSettings:
    def __init__(self):
        self.n_conv5_f = 3
        self.n_conv4_f = 5
        self.n_conv3_f = 7

        self.back_prop_mode = 'grad' # grad for gradient, deconv
        self.avg_pointcloud_width = 5

        self.thres_conv5 = 20# 15#20
        self.thres_conv4 = 5# 3#2
        self.thres_conv3 = 0.2# 0.1#0.2

        self.backprop_xy = 'sin' #'all'

        # avg back prop to image and do avg, max does filter level max xyz
        self.localize_layer = 'img' #'filter' #filter not functional

        self.top_filter = 'above' #'max'

        self.data_set = '103'

        self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]

        self.frame_list_conv4 = ["r2/left_palm"]

        self.frame_list_conv5 = [] # not implemented yet

    def get_name(self):
        name = '(' + str(len(self.frame_list_conv5)) + '-' + str(len(self.frame_list_conv4)) + '-' + str(len(self.frame_list_conv3)) + ')_(' + \
                str(self.n_conv5_f) + '-' + str(self.n_conv4_f) + '-' + str(self.n_conv3_f) + ')_' + \
                self.localize_layer + '_' + self.backprop_xy + '_' + self.data_set + '_' + self.back_prop_mode + '_' + \
                str(self.avg_pointcloud_width) + '_(' + str(self.thres_conv5) + '-' + str(self.thres_conv4) + '-' + str(self.thres_conv3) + ')_' + self.top_filter

        return name
