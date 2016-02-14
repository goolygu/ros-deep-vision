#! /usr/bin/env python
# -*- coding: utf-8

import sys
import os
import cv2
import numpy as np
import time
import StringIO
from threading import Lock

from misc import WithTimer
from numpy_cache import FIFOLimitedArrayCache
from app_base import BaseApp
from core import CodependentThread
from image_misc import norm01, norm01c, norm0255, tile_images_normalize, ensure_float01, tile_images_make_tiles, ensure_uint255_and_resize_to_fit, caffe_load_image, get_tiles_height_width
from image_misc import FormattedString, cv2_typeset_text, to_255, is_masked

from time import gmtime, strftime
from utils import DescriptorHandler, Descriptor, DataHandler
import caffe.io

from caffe_vis_app_state import CaffeVisAppState

import roslib
import rospy
from perception_msgs.srv import *
from geometry_msgs.msg import *

class CaffeProcThread(CodependentThread):
    '''Runs Caffe in separate thread.'''

    def __init__(self, net, state, loop_sleep, pause_after_keys, heartbeat_required):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.net = net
        self.input_dims = self.net.blobs['data'].data.shape[2:4]    # e.g. (227,227)
        self.state = state
        self.frames_processed_fwd = 0
        self.frames_processed_back = 0
        self.loop_sleep = loop_sleep
        self.pause_after_keys = pause_after_keys
        self.debug_level = 0
        self.descriptor = None
        self.descriptor_layer_1 = 'conv5'
        self.descriptor_layer_2 = 'conv4'
        self.descriptor_layers = ['conv5','conv4']
        self.net_input_image = None
        self.descriptor_handler = DescriptorHandler(self.state.settings.ros_dir + '/models/memory/', self.descriptor_layers)
        self.data_handler = DataHandler(self.state.settings.ros_dir)
        self.available_layer = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
        #['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
        # print "layers ", list(self.net._layer_names)

        s = rospy.Service('get_cnn_state', GetState, self.handle_get_cnn_state)

    def handle_get_cnn_state(self, req):
        resp = GetStateResponse()
        print "get req", req
        state = perception_msgs.msg.State()
        state.type = req.type
        state.name = req.name
        if req.name == "None":
            state.name = self.data_handler.save_net(self.net)
        else:
            print "not implemented yet"
        resp.state = state
        resp.pose = Pose()
        print "state", resp

        return resp

    def mask_out(self, data, mask):
        # print "data shape", data.shape
        dim = data.shape

        for y in range(dim[2]):
            for x in range(dim[3]):
                if is_masked((dim[2],dim[3]),(x,y),mask):
                    data[:,:,y,x] = 0

        return data

    def net_preproc_forward(self, img):
        assert img.shape == (227,227,3), 'img is wrong size'
        #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)
        data_blob = self.net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        output = self.net.forward(data=data_blob)
        return output

    def net_proc_forward_layer(self, img, mask):
        assert img.shape == (227,227,3), 'img is wrong size'
        #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)
        data_blob = self.net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        # print "mask", mask.shape

        for idx in range(len(self.available_layer)-1):
            output = self.net.forward(data=data_blob,start=self.available_layer[idx],end=self.available_layer[idx+1])
            if self.available_layer[idx].startswith("conv"):
                new_blob = self.net.blobs[self.available_layer[idx]].data
                new_blob.data = self.mask_out(self.net.blobs[self.available_layer[idx]].data, mask)
                self.net.blobs[self.available_layer[idx]] = new_blob
            # print output

        return output

    def save_descriptor(self):
        print 'save descriptor'
        # print type(self.net.blobs[self.descriptor_layer_1])
        # descriptor_1 = np.array(caffe.io.blobproto_to_array(self.net.blobs[self.descriptor_layer_1]))
        # descriptor_2 = np.array(caffe.io.blobproto_to_array(self.net.blobs[self.descriptor_layer_2]))
        # print descriptor
        self.time_name = strftime("%d-%m-%Y-%H:%M:%S", gmtime())

        desc = self.descriptor_handler.gen_descriptor(self.time_name, self.net.blobs)
        self.descriptor_handler.save(self.state.settings.ros_dir + '/models/memory/', desc)

        # np.save(self.state.settings.ros_dir + '/models/memory/' + self.time_name + "_" + self.descriptor_layer_1 + '.npy', descriptor_1)
        # np.save(self.state.settings.ros_dir + '/models/memory/' + self.time_name + "_" + self.descriptor_layer_2 + '.npy', descriptor_2)
        cv2.imwrite(self.state.settings.ros_dir + '/models/memory/' + self.time_name + '.jpg', self.net_input_image[:,:,::-1])

        self.state.save_descriptor = False

    def run(self):
        print 'CaffeProcThread.run called'
        frame = None
        mask = None
        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:
                    #print 'CaffeProcThread.run: quit is True'
                    #print self.state.quit
                    break

                #print 'CaffeProcThread.run: caffe_net_state is:', self.state.caffe_net_state

                #print 'CaffeProcThread.run loop: next_frame: %s, caffe_net_state: %s, back_enabled: %s' % (
                #    'None' if self.state.next_frame is None else 'Avail',
                #    self.state.caffe_net_state,
                #    self.state.back_enabled)

                frame = None
                mask = None
                run_fwd = False
                run_back = False
                if self.state.caffe_net_state == 'free' and time.time() - self.state.last_key_at > self.pause_after_keys:
                    frame = self.state.next_frame
                    mask = self.state.mask
                    self.state.next_frame = None
                    back_enabled = self.state.back_enabled
                    back_mode = self.state.back_mode
                    back_stale = self.state.back_stale
                    #state_layer = self.state.layer
                    #selected_unit = self.state.selected_unit
                    backprop_layer = self.state.backprop_layer
                    backprop_unit = self.state.backprop_unit

                    # Forward should be run for every new frame
                    run_fwd = (frame is not None)
                    # Backward should be run if back_enabled and (there was a new frame OR back is stale (new backprop layer/unit selected))
                    run_back = (back_enabled and (run_fwd or back_stale))
                    self.state.caffe_net_state = 'proc' if (run_fwd or run_back) else 'free'

            #print 'run_fwd,run_back =', run_fwd, run_back

            if run_fwd:
                #print 'TIMING:, processing frame'
                self.frames_processed_fwd += 1
                self.net_input_image = cv2.resize(frame, self.input_dims)
                with WithTimer('CaffeProcThread:forward', quiet = self.debug_level < 1):
                    print "run forward layer"
                    self.net_proc_forward_layer(self.net_input_image, mask)
                    # self.net_preproc_forward(self.net_input_image)

            if self.state.save_descriptor:
                self.save_descriptor()

            # switch descriptor for match and back prop
            if self.descriptor is None or self.state.next_descriptor:
                    print 'load descriptor'
                    self.descriptor = self.descriptor_handler.get_next()
                    self.state.next_descriptor = False


            if self.state.compare_descriptor:
                # find
                print 'compare'
                desc_current = self.descriptor_handler.gen_descriptor('current', self.net.blobs)
                match_file = self.descriptor_handler.get_max_match(desc_current)
                print 'match: ' + match_file
                self.state.compare_descriptor = False

            if run_back:
                print "run backward"
                # Match to saved descriptor
                if self.state.match_descriptor:
                    print '*'
                    diffs = self.net.blobs[self.descriptor_layer_1].diff * 0

                    # zero all diffs if doesn't match
                    print "shape ", self.net.blobs[self.descriptor_layer_1].data.shape
                    for unit, response in enumerate(self.net.blobs[self.descriptor_layer_1].data[0]):
                        if response.max() > 0 and abs(response.max() - self.descriptor.get_sig_list()[0][0][unit].max())/response.max() < 0.2:
                            diffs[0][unit] = self.net.blobs[self.descriptor_layer_1].data[0][unit]

                    assert back_mode in ('grad', 'deconv')
                    if back_mode == 'grad':
                        with WithTimer('CaffeProcThread:backward', quiet = self.debug_level < 1):
                            #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                            self.net.backward_from_layer(self.descriptor_layer_1, diffs, zero_higher = True)
                    else:
                        with WithTimer('CaffeProcThread:deconv', quiet = self.debug_level < 1):
                            #print '**** Doing deconv with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                            self.net.deconv_from_layer(self.descriptor_layer_1, diffs, zero_higher = True)

                    with self.state.lock:
                        self.state.back_stale = False

                # Filter when back propagating
                elif self.state.backprop_filter:
                    print "run_back"
                    # print backprop_layer
                    start_layer_idx = self.available_layer.index(backprop_layer)
                    idx = start_layer_idx
                    for current_layer in list(reversed(self.available_layer[0:start_layer_idx+1])):

                        diffs = self.net.blobs[current_layer].diff * 0
                        max_response = self.net.blobs[current_layer].data[0].max()
                        for unit, response in enumerate(self.net.blobs[current_layer].data[0]):
                            if response.max() > max_response * 0.6:
                                diffs[0][unit] = self.net.blobs[current_layer].data[0,unit]


                        assert back_mode in ('grad', 'deconv')
                        if back_mode == 'grad':
                            with WithTimer('CaffeProcThread:backward', quiet = self.debug_level < 1):
                                #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                                self.net.backward_from_to_layer(current_layer, diffs, self.available_layer[idx-1], zero_higher = (idx == start_layer_idx))
                        # else:
                        #     with WithTimer('CaffeProcThread:deconv', quiet = self.debug_level < 1):
                        #         #print '**** Doing deconv with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                        #         self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)
                        idx -= 1
                    with self.state.lock:
                        self.state.back_stale = False

                # original approach
                else:
                    diffs = self.net.blobs[backprop_layer].diff * 0
                    diffs[0][backprop_unit] = self.net.blobs[backprop_layer].data[0,backprop_unit]

                    assert back_mode in ('grad', 'deconv')
                    if back_mode == 'grad':
                        with WithTimer('CaffeProcThread:backward', quiet = self.debug_level < 1):
                            #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
                    else:
                        with WithTimer('CaffeProcThread:deconv', quiet = self.debug_level < 1):
                            #print '**** Doing deconv with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)

                    with self.state.lock:
                        self.state.back_stale = False

            if run_fwd or run_back:
                with self.state.lock:
                    self.state.caffe_net_state = 'free'
                    self.state.drawing_stale = True
            else:
                time.sleep(self.loop_sleep)
            time.sleep(0.1)
        print 'CaffeProcThread.run: finished'
        print 'CaffeProcThread.run: processed %d frames fwd, %d frames back' % (self.frames_processed_fwd, self.frames_processed_back)
