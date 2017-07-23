#! /usr/bin/env python
class DataSettings:
    def __init__(self, tbp):

        if tbp:
            self.n_conv5_f = 3#0#3#5
            self.n_conv4_f = 3#3#9#3
            self.n_conv3_f = 3#3#27#3
        else:
            self.n_conv5_f = 0#3#0#3#5
            self.n_conv4_f = 16#9#3#9#3
            self.n_conv3_f = 64#27#3#27#3

        self.back_prop_mode = 'grad' #'deconv'# grad for gradient, deconv
        self.xyz_back_prop = 'grad' #'deconv'
        self.filters = 'spread'
        self.avg_pointcloud_width = 5

        self.thres_conv5 = 0#20# 15#20
        self.thres_conv4 = 0#5# 3#2
        self.thres_conv3 = 0#0.2# 0.1#0.2





        self.backprop_xy = 'all' #'sin' #

        # avg back prop to image and do avg, max does filter level max xyz
        # self.localize_layer = 'img' #'filter' #filter not functional

        self.top_filter = 'max'#'above' #'max'

        self.data_set = '103'

        self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]

        self.frame_list_conv4 = ["r2/left_palm"]

        self.frame_list_conv5 = [] # not implemented yet

        ###### Testing related


        self.filter_test = 'top' #'all' #
        if tbp:
            self.conv5_top = 5#0#3#3
            self.conv4_top = 3#9#3
            self.conv3_top = 2#27#3
        else:
            self.conv5_top = 0#3#0#3#3
            self.conv4_top = 16#3#9#3
            self.conv3_top = 64#3#27#3

        self.thres_conv5_test = 0# 15#20
        self.thres_conv4_test = 0#2#5# 3#2
        self.thres_conv3_test = 0#0.2#0.2# 0.1#0.2

        self.pointcloud = 'seg' # seg
        self.mask = 'mask'
        self.evaluate = 'full' # 'case'
        # how is the grasp point determined from a distribution
        self.dist_to_grasp_point = "densepoint" #"density" "center"

    def get_name(self):
        name = '(' + str(len(self.frame_list_conv5)) + '-' + str(len(self.frame_list_conv4)) + '-' + str(len(self.frame_list_conv3)) + ')_(' + \
                str(self.n_conv5_f) + '-' + str(self.n_conv4_f) + '-' + str(self.n_conv3_f) + ')_' + \
                self.backprop_xy + '_' + self.data_set + '_' + self.back_prop_mode + '_' + \
                str(self.avg_pointcloud_width) + '_(' + str(self.thres_conv5) + '-' + str(self.thres_conv4) + '-' + str(self.thres_conv3) + ')_' + self.top_filter

        return name

    def get_test_name(self):
        name =  self.filter_test + '_' + \
                '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + ')_' + \
                '(' + str(self.thres_conv5_test) + '-' + str(self.thres_conv4_test) + '-' + str(self.thres_conv3_test) + ')_' + \
                self.evaluate + '_' + self.mask + '_' + self.pointcloud + "_" + self.dist_to_grasp_point

        return name
