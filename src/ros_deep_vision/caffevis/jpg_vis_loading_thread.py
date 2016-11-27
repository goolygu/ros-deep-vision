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

from caffe_vis_app_state import CaffeVisAppState

def crop_to_corner(img, corner, small_padding = 1, large_padding = 2):
    '''Given an large image consisting of 3x3 squares with small_padding padding concatenated into a 2x2 grid with large_padding padding, return one of the four corners (0, 1, 2, 3)'''
    assert corner in (0,1,2,3), 'specify corner 0, 1, 2, or 3'
    assert img.shape[0] == img.shape[1], 'img is not square'
    assert img.shape[0] % 2 == 0, 'even number of pixels assumption violated'
    half_size = img.shape[0]/2
    big_ii = 0 if corner in (0,1) else 1
    big_jj = 0 if corner in (0,2) else 1
    tp = small_padding + large_padding
    #tp = 0
    return img[big_ii*half_size+tp:(big_ii+1)*half_size-tp,
               big_jj*half_size+tp:(big_jj+1)*half_size-tp]

    #image_pixels = img.shape[0] - small_padding * 12 - large_padding * 4
    #assert image_pixels % 6 == 0, 'math error'
    #small_image_size = image_pixels / 6

class JPGVisLoadingThread(CodependentThread):
    '''Loads JPGs necessary for caffevis_jpgvis pane in separate
    thread and inserts them into the cache.
    '''

    def __init__(self, settings, state, cache, loop_sleep, heartbeat_required):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.settings = settings
        self.state = state
        self.cache = cache
        self.loop_sleep = loop_sleep
        self.debug_level = 0

    def run(self):
        print 'JPGVisLoadingThread.run called'

        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:
                    break

                #print 'JPGVisLoadingThread.run: caffe_net_state is:', self.state.caffe_net_state
                #print 'JPGVisLoadingThread.run loop: next_frame: %s, caffe_net_state: %s, back_enabled: %s' % (
                #    'None' if self.state.next_frame is None else 'Avail',
                #    self.state.caffe_net_state,
                #    self.state.back_enabled)

                jpgvis_to_load_key = self.state.jpgvis_to_load_key

            if jpgvis_to_load_key is None:
                time.sleep(self.loop_sleep)
                continue

            state_layer, state_selected_unit, data_shape = jpgvis_to_load_key

            # Load three images:
            images = [None] * 3

            # Reize each component images only using one direction as
            # a constraint. This is straightfowrad but could be very
            # wasteful (making an image much larger then much smaller)
            # if the proportions of the stacked image are very
            # different from the proportions of the data pane.
            #resize_shape = (None, data_shape[1]) if self.settings.caffevis_jpgvis_stack_vert else (data_shape[0], None)
            # As a heuristic, instead just assume the three images are of the same shape.
            if self.settings.caffevis_jpgvis_stack_vert:
                resize_shape = (data_shape[0]/3, data_shape[1])
            else:
                resize_shape = (data_shape[0], data_shape[1]/3)

            # 0. e.g. regularized_opt/conv1/conv1_0037_montage.jpg
            jpg_path = os.path.join(self.settings.caffevis_unit_jpg_dir,
                                    'regularized_opt',
                                    state_layer,
                                    '%s_%04d_montage.jpg' % (state_layer, state_selected_unit))
            try:
                img = caffe_load_image(jpg_path, color = True)
                img_corner = crop_to_corner(img, 2)
                images[0] = ensure_uint255_and_resize_to_fit(img_corner, resize_shape)
            except IOError:
                pass

            # 1. e.g. max_im/conv1/conv1_0037.jpg
            jpg_path = os.path.join(self.settings.caffevis_unit_jpg_dir,
                                    'max_im',
                                    state_layer,
                                    '%s_%04d.jpg' % (state_layer, state_selected_unit))
            try:
                img = caffe_load_image(jpg_path, color = True)
                images[1] = ensure_uint255_and_resize_to_fit(img, resize_shape)
            except IOError:
                pass

            # 2. e.g. max_deconv/conv1/conv1_0037.jpg
            try:
                jpg_path = os.path.join(self.settings.caffevis_unit_jpg_dir,
                                        'max_deconv',
                                        state_layer,
                                        '%s_%04d.jpg' % (state_layer, state_selected_unit))
                img = caffe_load_image(jpg_path, color = True)
                images[2] = ensure_uint255_and_resize_to_fit(img, resize_shape)
            except IOError:
                pass

            # Prune images that were not found:
            images = [im for im in images if im is not None]

            # Stack together
            if len(images) > 0:
                #print 'Stacking:', [im.shape for im in images]
                stack_axis = 0 if self.settings.caffevis_jpgvis_stack_vert else 1
                img_stacked = np.concatenate(images, axis = stack_axis)
                #print 'Stacked:', img_stacked.shape
                img_resize = ensure_uint255_and_resize_to_fit(img_stacked, data_shape)
                #print 'Resized:', img_resize.shape
            else:
                img_resize = np.zeros(shape=(0,))   # Sentinal value when image is not found.

            self.cache.set(jpgvis_to_load_key, img_resize)

            with self.state.lock:
                self.state.jpgvis_to_load_key = None
                self.state.drawing_stale = True

        print 'JPGVisLoadingThread.run: finished'
