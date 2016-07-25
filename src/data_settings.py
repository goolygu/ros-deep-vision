#! /usr/bin/env python
class DataSettings:
    def __init__(self, tbp):


        if tbp:
            self.tbp_str = 'tbp'
            self.n_conv5_f = 3#0#3#5
            self.n_conv4_f = 3#3#9#3
            self.n_conv3_f = 3#3#27#3
        else:
            self.tbp_str = 'notbp'
            self.n_conv5_f = 0#3#0#3#5
            self.n_conv4_f = 9#9#3#9#3
            self.n_conv3_f = 27#27#3#27#3

        self.back_prop_mode = 'grad' # grad for gradient, deconv
        self.avg_pointcloud_width = 5

        self.thres_conv5 = 0#20# 15#20
        self.thres_conv4 = 0#5# 3#2
        self.thres_conv3 = 0#0.2# 0.1#0.2

        self.location_layer = "image" #"mid" #

        self.backprop_xy = 'sin' #'all' #

        # avg back prop to image and do avg, max does filter level max xyz
        # self.localize_layer = 'img' #'filter' #filter not functional

        self.top_filter = 'maxlog'#'above' #'max'

        self.dataset = 'set3'

        self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]

        self.frame_list_conv4 = ["r2/left_palm"]

        self.frame_list_conv5 = [] # not implemented yet

        self.input_width = 260

        self.img_src_loc ="relu" #"absolute"

        ###### Testing related


        self.filter_test = 'top' #'all' #
        if tbp:
            self.conv5_top = 3#0#3#3
            self.conv4_top = 3#9#3
            self.conv3_top = 3#27#3
        else:
            self.conv5_top = 0#3#0#3#3
            self.conv4_top = 9#3#9#3
            self.conv3_top = 27#3#27#3

        self.thres_conv5_test = 0# 15#20
        self.thres_conv4_test = 0#2#5# 3#2
        self.thres_conv3_test = 0#0.2#0.2# 0.1#0.2

        self.pointcloud = 'seg' # seg
        self.mask = 'mask'
        self.evaluate = 'case' # 'full' #
        # how is the grasp point determined from a distribution
        self.dist_to_grasp_point = "densepoint" #"mean" #"filterweightmean"#"weightmean" #"densepoint" #"density" # #

        self.filter_low_n = -1

    def get_name(self):
        name =  self.tbp_str + "_" + '(' + str(len(self.frame_list_conv5)) + '-' + str(len(self.frame_list_conv4)) + '-' + str(len(self.frame_list_conv3)) + ')_(' + \
                str(self.n_conv5_f) + '-' + str(self.n_conv4_f) + '-' + str(self.n_conv3_f) + ')_' + \
                self.backprop_xy + '_' + self.dataset + '_' + self.back_prop_mode + '_' + \
                str(self.avg_pointcloud_width) + '_(' + str(self.thres_conv5) + '-' + str(self.thres_conv4) + '-' + str(self.thres_conv3) + ')_' + \
                self.top_filter + '_' + self.location_layer + "_" + str(self.input_width) + "_" + self.img_src_loc

        return name

    def get_test_name(self):
        name =  self.filter_test + '_' + \
                '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + ')_' + \
                '(' + str(self.thres_conv5_test) + '-' + str(self.thres_conv4_test) + '-' + str(self.thres_conv3_test) + ')_' + \
                self.evaluate + '_' + self.mask + '_' + self.pointcloud + "_" + self.dist_to_grasp_point + "_" + str(self.filter_low_n)

        return name
