#! /usr/bin/env python
# This is the config file

class DataSettings:
    def __init__(self, case="tbp"):
        self.tbp = True
        self.tbp_str = 'tbp'
        # backprop with gradient or deconvolution
        self.back_prop_mode = 'grad' # grad for gradient, deconv

        # how large of area should the mean xyz location be based on on the point cloud
        self.avg_pointcloud_width = 5

        self.thres_conv5 = 0
        self.thres_conv4 = 0
        self.thres_conv3 = 0
        self.thres_conv2 = 0

        # whether to backprop all the way to the image to locate
        self.location_layer = "image" #"mid" #

        # only backprop a single filter
        self.backprop_xy = 'sin' #'all' #

        # avg back prop to image and do avg, max does filter level max xyz
        # self.localize_layer = 'img' #'filter' #filter not functional

        self.top_filter = 'maxlog'#'above' #'max'

        self.input_width = 260

        # xy location on image plane based on absolute derivative or derivative above 0
        self.img_src_loc = "relu" #"absolute"#

        # use mask centering if you want the image to be centered based on the mask. Use on cluttered dataset.
        self.mask_centering = True#False

        ###### Pose test
        self.filters = 'top'#'spread'#
        self.cnn_pose_state = 'xyz'#'reldist' #
        self.square = 'inc'#'dec'#
        self.cnn_pose_state_match = 'full'#'local'#'top'#
        self.cloud_gap = 'inpaint'#'nan'#
        self.similarity = 'heavytail'#'gaussian'#'clean'#'custom'#'L1'

        ###### Testing related

        self.filter_test = 'top' #'all' #

        self.thres_conv5_test = 0
        self.thres_conv4_test = 0
        self.thres_conv3_test = 0
        self.thres_conv2_test = 0

        # use segmented point cloud
        self.pointcloud = 'seg' # seg
        # use masked forward pass
        self.mask = 'mask'
        # evaluate on the same case or all
        self.evaluate = 'case' # 'full' #
        # how is the grasp point determined from a distribution
        self.dist_to_grasp_point = "mean"#"weightmean" #"densepoint" #"filterweightmean"#"densepoint" #"density"
        # whether restrict to the same parent when filtering
        self.filter_same_parent = False
        # whether filter out high variance features, number of features kept, -1 if no filtering
        self.filter_low_n = -1
        self.tbp_test_str = ""
        self.tbp_test = True

        # comparing 5 different cases with and without targeted backpropagation
        # note the following may overwrite settings above
        if case == "tbp" or case == "notbp-test":

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
            if case == "notbp-test":
                self.tbp_test = False
                self.tbp_test_str = '_notbptest'

        elif case == "single-conv5":

            self.n_conv5_f = 5
            self.n_conv4_f = 0
            self.n_conv3_f = 0
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = []
            self.frame_list_conv4 = []
            self.frame_list_conv5 = ["r2/left_thumb_tip","r2/left_index_tip","r2/left_palm"]
            self.conv5_top = 1
            self.conv4_top = 0
            self.conv3_top = 0
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = -1
        elif case == "notbp-conv5":

            self.n_conv5_f = 125
            self.n_conv4_f = 0
            self.n_conv3_f = 0
            self.n_conv2_f = 0
            self.frame_list_conv2 = []
            self.frame_list_conv3 = []
            self.frame_list_conv4 = []
            self.frame_list_conv5 = ["r2/left_thumb_tip","r2/left_index_tip","r2/left_palm"]
            self.conv5_top = 125
            self.conv4_top = 0
            self.conv3_top = 0
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = 15
        elif case == "notbp-train":
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
            self.conv5_top = 0
            self.conv4_top = 25
            self.conv3_top = 125
            self.conv2_top = 0
            self.dist_to_grasp_point = "weightmean"#"densepoint" #"weightdensepoint"#
            self.filter_same_parent = False
            self.filter_low_n = 15
        # This is for pose test on the washinton dataset
        elif case == "pose_test":
            self.conv5_top = 30
            self.conv4_top = 5
            self.conv3_top = 0
            self.conv2_top = 0
        # This is for manipulation experiments on R2
        elif case == "cnn_features":
            self.conv5_top = 10
            self.conv4_top = 5
            self.conv3_top = 2
            self.conv2_top = 0
        else:
            print "ERROR, no such case", case

        if self.filter_same_parent:
            self.filter_same_parent_str = "_sameparent"
        else:
            self.filter_same_parent_str = ""

    def get_name(self):
        name =  self.tbp_str + "_" + '(' + str(len(self.frame_list_conv5)) + '-' + str(len(self.frame_list_conv4)) + '-' + str(len(self.frame_list_conv3)) + '-' + str(len(self.frame_list_conv2)) + ')_(' + \
                str(self.n_conv5_f) + '-' + str(self.n_conv4_f) + '-' + str(self.n_conv3_f) + '-' + str(self.n_conv2_f) + ')_' + \
                self.backprop_xy + '_' + self.back_prop_mode + '_' + \
                str(self.avg_pointcloud_width) + '_(' + str(self.thres_conv5) + '-' + str(self.thres_conv4) + '-' + str(self.thres_conv3) + '-' + str(self.thres_conv2) + ')_' + \
                self.top_filter + '_' + self.location_layer + "_" + str(self.input_width) + "_" + self.img_src_loc

        return name

    def get_pose_state_name(self):
        name = '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + '-' + str(self.conv2_top) + ')_' + "_" + self.filters + "_" + \
                "_" + self.square + "_" + self.cloud_gap
        return name

    def get_pose_state_test_name(self):
        name = self.cnn_pose_state + "_" + self.cnn_pose_state_match + "_" + self.similarity
        return name

    def get_test_name(self):
        name =  self.filter_test + '_' + \
                '(' + str(self.conv5_top) + '-' + str(self.conv4_top) + '-' + str(self.conv3_top) + '-' + str(self.conv2_top) + ')_' + \
                '(' + str(self.thres_conv5_test) + '-' + str(self.thres_conv4_test) + '-' + str(self.thres_conv3_test) + '-' + str(self.thres_conv2_test) + ')_' + \
                self.evaluate + '_' + self.mask + '_' + self.pointcloud + "_" + self.dist_to_grasp_point + "_" + str(self.filter_low_n) + \
                self.filter_same_parent_str + self.tbp_test_str

        return name
