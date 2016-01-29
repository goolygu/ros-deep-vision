#! /usr/bin/env python
import numpy as np
import re
import os
import caffe.io

class Descriptor:

    def __init__(self, tag, sig_list):
        self.tag = tag
        self.sig_list = sig_list
        self.threshold = 400
        self.sig_list_binary = self.cal_sig_list_binary()

    def get_sig_list(self):
        return self.sig_list

    def cal_sig_list_binary(self):
        sig_list_binary = []
        for sig in self.sig_list:
            sig_copy = np.copy(sig)
            sig_bin = np.sum(sig_copy, axis=(-2,-1))
            sig_bin[sig_bin < self.threshold] = 0
            sig_bin[sig_bin >= self.threshold] = 1
            sig_list_binary.append(sig_bin)
        return sig_list_binary

    # def get_sig_list_binary(self):
    #     sig_list_binary = []
    #     for sig in self.sig_list:
    #         sig_bin = np.copy(sig)
    #         sig_bin[sig_bin < self.threshold] = 0
    #         sig_bin[sig_bin >= self.threshold] = 1
    #         sig_list_binary.append(sig_bin)
    #     return sig_list_binary

class DescriptorHandler:

    def __init__(self, dir, descriptor_layers):
        self.count = 0;
        self.file_tag_list = []
        self.dir = dir
        self.descriptor_layers = descriptor_layers
        self.descriptor_dict = {}

        self.init_tag_list()
        self.load_all()

    def init_tag_list(self):
        match_flags = re.IGNORECASE
        for filename in os.listdir(self.dir):
            if re.match('.*\.(jpg|jpeg|png)$', filename, match_flags):
                self.file_tag_list.append(filename.split('.')[0])

            # if re.match('.*_rgb\.(jpg|jpeg|png)$', filename, match_flags):
            #     self.file_tag_list.append(filename.split('_rgb')[0])

    def load_next(self):

        idx = (self.count) % len(self.file_tag_list)

        # print 'load ' + self.file_tag_list[idx]
        sig_1 = np.load(self.dir + self.file_tag_list[idx] + '_' + self.descriptor_layers[0] +'.npy' )
        sig_2 = np.load(self.dir + self.file_tag_list[idx] + '_' + self.descriptor_layers[1] +'.npy' )
        self.count += 1
        return Descriptor(self.file_tag_list[idx], [sig_1, sig_2])

    def load_all(self):
        self.count = 0
        while self.count < len(self.file_tag_list):
            desc = self.load_next()
            self.descriptor_dict[desc.tag] = desc

    def gen_descriptor(self, tag, blobs):
        sig_list = []
        for i, layer_name in enumerate(self.descriptor_layers):
            sig = np.array(caffe.io.blobproto_to_array(blobs[layer_name]))
            sig_list.append(sig)

        return Descriptor(tag, sig_list)

    def save(self, directory, desc):
        for i, sig in enumerate(desc.get_sig_list()):
            np.save(directory + desc.tag + "_" + self.descriptor_layers[i] + '.npy', sig)


    def get_next(self):

        idx = (self.count) % len(self.file_tag_list)
        desc = self.descriptor_dict[self.file_tag_list[idx]]
        self.count += 1
        print desc.tag
        return desc

    def get_max_match(self, desc):
        max_sim = 0
        max_key = ""
        for key in self.descriptor_dict:
            similarity = self.get_signature_similarity(desc, self.descriptor_dict[key])
            if similarity > max_sim:
                max_sim = similarity
                max_key = key
            # print "key ", key, " similarity ", similarity
        return max_key
        # print "max: ", max_key

    def get_signature_similarity(self, desc, desc_model):

        sig = desc.sig_list_binary
        sig_model = desc_model.sig_list_binary
        percentage = 1.0

        for i, layer in enumerate(self.descriptor_layers):
            p= self.get_match_percentage(sig[i], sig_model[i])
            # print "layer ", i, " percentage ", p
            percentage *= p

        return percentage

    def get_match_percentage(self, sig, sig_model):
        match = np.zeros(sig.shape,dtype=bool)
        match[sig == sig_model] = 1
        return match.sum()/float(match.size)

        # match = np.ones(sig.shape,dtype=bool)
        # match[(sig == 0) & (sig_model == 1)] = 0
        # return match.sum()/float(match.size)
