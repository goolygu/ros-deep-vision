#! /usr/bin/env python
class DataSettings:
    def __init__(self):

        self.tbp = True
        if self.tbp:
            self.tbp_str = 'tbp'
            self.n_conv5_f = 1#0#3#5
            self.n_conv4_f = 0#3#9#3
            self.n_conv3_f = 0#3#27#3
            self.n_conv2_f = 0
        else:
            self.tbp_str = 'notbp'
            self.n_conv5_f = 0#3#0#3#5
            self.n_conv4_f = 9#9#3#9#3
            self.n_conv3_f = 18#27#3#27#3
            self.n_conv2_f = 0

        self.back_prop_mode = 'grad' # grad for gradient, deconv
        self.avg_pointcloud_width = 5

        self.thres_conv5 = 0#20# 15#20
        self.thres_conv4 = 0#5# 3#2
        self.thres_conv3 = 0#0.2# 0.1#0.2
        self.thres_conv2 = 0

        self.location_layer = "image" #"mid" #

        self.backprop_xy = 'sin' #'all' #

        # avg back prop to image and do avg, max does filter level max xyz
        # self.localize_layer = 'img' #'filter' #filter not functional

        self.top_filter = 'maxlog'#'above' #'max'

        self.dataset = 'set7'

        self.frame_list_conv2 = []
        self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]
        self.frame_list_conv4 = ["r2/left_palm"]
        self.frame_list_conv5 = []

        self.input_width = 260

        self.img_src_loc = "relu" #"absolute"#

        ###### state only not saved
        self.filters = 'top'#'spread'#
        self.cnn_pose_state = 'xyz'#'reldist' #
        self.square = 'inc'#'dec'#
        self.cnn_pose_state_match = 'full'#'local'#'top'#
        self.cloud_gap = 'inpaint'#'nan'#
        self.similarity = 'heavytail'#'gaussian'#'clean'#'custom'#'L1'
        ###### Testing related


        self.filter_test = 'top' #'all' #
        if self.tbp:
            self.conv5_top = 1#0#3#3
            self.conv4_top = 0#9#3
            self.conv3_top = 0#27#3
            self.conv2_top = 0
        else:
            self.conv5_top = 0#3#0#3#3
            self.conv4_top = 9#3#9#3
            self.conv3_top = 18#3#27#3
            self.conv2_top = 0

        self.thres_conv5_test = 0# 15#20
        self.thres_conv4_test = 0#2#5# 3#2
        self.thres_conv3_test = 0#0.2#0.2# 0.1#0.2
        self.thres_conv2_test = 0

        self.pointcloud = 'seg' # seg
        self.mask = 'mask'
        self.evaluate = 'case' # 'full' #
        # how is the grasp point determined from a distribution
        self.dist_to_grasp_point = "mean"#"weightmean" #"densepoint" #"filterweightmean"#"densepoint" #"density" # #
        self.filter_same_parent = False

        self.filter_low_n = -1
        self.tbp_test_str = ""
        self.tbp_test = True


        case = 1
        # over write 3 cases for comparison
        if case == 1 or case == 5:
            self.tbp = True
            self.tbp_str = 'tbp'
            self.n_conv5_f = 5
            self.n_conv4_f = 5
            self.n_conv3_f = 5
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]
            self.frame_list_conv4 = ["r2/left_palm"]
            self.frame_list_conv5 = []
            self.conv5_top = self.n_conv5_f
            self.conv4_top = self.n_conv4_f
            self.conv3_top = self.n_conv3_f
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = True
            self.filter_low_n = 15
            if case == 5:
                self.tbp_test = False
                self.tbp_test_str = '_notbptest'

        elif case == 2:
            self.tbp = True
            self.tbp_str = 'tbp'
            self.n_conv5_f = 5
            self.n_conv4_f = 0
            self.n_conv3_f = 0
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = []
            self.frame_list_conv4 = []
            self.frame_list_conv5 = ["r2/left_thumb_tip","r2/left_index_tip","r2/left_palm"]
            self.conv5_top = 1#0#3#3
            self.conv4_top = 0#9#3
            self.conv3_top = 0#27#3
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = -1
        elif case == 3:
            self.tbp = True
            self.tbp_str = 'tbp'
            self.n_conv5_f = 125
            self.n_conv4_f = 0
            self.n_conv3_f = 0
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = []
            self.frame_list_conv4 = []
            self.frame_list_conv5 = ["r2/left_thumb_tip","r2/left_index_tip","r2/left_palm"]
            self.conv5_top = 125#0#3#3
            self.conv4_top = 0#9#3
            self.conv3_top = 0#27#3
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = 15
        elif case == 4:
            self.tbp = False
            self.tbp_str = 'notbp'
            self.n_conv5_f = 0
            self.n_conv4_f = 25
            self.n_conv3_f = 125
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = ["r2/left_thumb_tip","r2/left_index_tip"]
            self.frame_list_conv4 = ["r2/left_palm"]
            self.frame_list_conv5 = []
            self.conv5_top = 0#0#3#3
            self.conv4_top = 25#9#3
            self.conv3_top = 125#27#3
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = 15
        elif case == 20:
            self.conv5_top = 30
            self.conv4_top = 5
            self.conv3_top = 0
            self.conv2_top = 0


        if self.filter_same_parent:
            self.filter_same_parent_str = "_sameparent"
        else:
            self.filter_same_parent_str = ""

    def get_name(self):
        name =  self.tbp_str + "_" + '(' + str(len(self.frame_list_conv5)) + '-' + str(len(self.frame_list_conv4)) + '-' + str(len(self.frame_list_conv3)) + '-' + str(len(self.frame_list_conv2)) + ')_(' + \
                str(self.n_conv5_f) + '-' + str(self.n_conv4_f) + '-' + str(self.n_conv3_f) + '-' + str(self.n_conv2_f) + ')_' + \
                self.backprop_xy + '_' + self.dataset + '_' + self.back_prop_mode + '_' + \
                str(self.avg_pointcloud_width) + '_(' + str(self.thres_conv5) + '-' + str(self.thres_conv4) + '-' + str(self.thres_conv3) + '-' + str(self.thres_conv2) + ')_' + \
                self.top_filter + '_' + self.location_layer + "_" + str(self.input_width) + "_" + self.img_src_loc

        return name

    def get_pose_state_name(self):
        name = '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + '-' + str(self.conv2_top) + ')_' + "_" + self.filters + "_" + \
                "_" + self.square + "_" + self.cloud_gap
                #self.cnn_pose_state \
        return name

    def get_pose_state_test_name(self):
        name = self.cnn_pose_state + "_" + self.cnn_pose_state_match + "_" + self.similarity + "_case9"
        return name

    def get_test_name(self):
        name =  self.filter_test + '_' + \
                '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + '-' + str(self.conv2_top) + ')_' + \
                '(' + str(self.thres_conv5_test) + '-' + str(self.thres_conv4_test) + '-' + str(self.thres_conv3_test) + '-' + str(self.thres_conv2_test) + ')_' + \
                self.evaluate + '_' + self.mask + '_' + self.pointcloud + "_" + self.dist_to_grasp_point + "_" + str(self.filter_low_n) + \
                self.filter_same_parent_str + self.tbp_test_str

        return name
