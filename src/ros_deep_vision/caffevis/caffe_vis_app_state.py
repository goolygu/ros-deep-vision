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

class CaffeVisAppState(object):
    '''State of CaffeVis app.'''

    def __init__(self, net, settings, bindings):
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings
        self._layers = net.blobs.keys()
        self._layers = self._layers[1:]  # chop off data layer
        self.layer_boost_indiv_choices = self.settings.caffevis_boost_indiv_choices   # 0-1, 0 is noop
        self.layer_boost_gamma_choices = self.settings.caffevis_boost_gamma_choices   # 0-inf, 1 is noop
        self.caffe_net_state = 'free'     # 'free', 'proc', or 'draw'
        # Which layer and unit (or channel) to use for backprop
        self.tiles_height_width = (1,1)   # Before any update
        self.tiles_number = 1
        self.extra_msg = ''
        self.back_stale = True       # back becomes stale whenever the last back diffs were not computed using the current backprop unit and method (bprop or deconv)
        self.next_frame = None
        self.jpgvis_to_load_key = None
        self.last_key_at = 0
        self.quit = False
        self.mask = None

        self._reset_user_state()

    def _reset_user_state(self):
        self.layer_idx = 0
        self.layer = self._layers[0]
        self.layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
        self.layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
        self.cursor_area = 'top'   # 'top' or 'bottom'
        self.selected_unit = 0
        self.backprop_layer = self.layer
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.back_enabled = False
        self.back_mode = 'grad'      # 'grad' or 'deconv'
        self.back_filt_mode = 'raw'  # 'raw', 'gray', 'norm', 'normblur'
        self.pattern_mode = False    # Whether or not to show desired patterns instead of activations in layers pane
        self.layers_pane_zoom_mode = 0       # 0: off, 1: zoom selected (and show pref in small pane), 2: zoom backprop
        self.layers_show_back = False   # False: show forward activations. True: show backward diffs
        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs
        self.drawing_stale = True
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]
        self.backprop_filter = False
        self.save_descriptor = False
        self.match_descriptor = False
        self.next_descriptor = False
        self.compare_descriptor = False

    def handle_key(self, key):
        #print 'Ignoring key:', key
        if key == -1:
            return key

        with self.lock:
            key_handled = True
            self.last_key_at = time.time()
            tag = self.bindings.get_tag(key)
            if tag == 'reset_state':
                self._reset_user_state()
            elif tag == 'sel_layer_left':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = max(0, self.layer_idx - 1)
                self.layer = self._layers[self.layer_idx]
                self._ensure_valid_selected()
            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + 1)
                self.layer = self._layers[self.layer_idx]
                self._ensure_valid_selected()

            elif tag == 'sel_left':
                self.move_selection('left')
            elif tag == 'sel_right':
                self.move_selection('right')
            elif tag == 'sel_down':
                self.move_selection('down')
            elif tag == 'sel_up':
                self.move_selection('up')

            elif tag == 'sel_left_fast':
                self.move_selection('left', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_right_fast':
                self.move_selection('right', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_down_fast':
                self.move_selection('down', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_up_fast':
                self.move_selection('up', self.settings.caffevis_fast_move_dist)

            elif tag == 'boost_individual':
                self.layer_boost_indiv_idx = (self.layer_boost_indiv_idx + 1) % len(self.layer_boost_indiv_choices)
                self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
            elif tag == 'boost_gamma':
                self.layer_boost_gamma_idx = (self.layer_boost_gamma_idx + 1) % len(self.layer_boost_gamma_choices)
                self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
            elif tag == 'pattern_mode':
                self.pattern_mode = not self.pattern_mode
            elif tag == 'show_back':
                # If in pattern mode: switch to fwd/back. Else toggle fwd/back mode
                if self.pattern_mode:
                    self.pattern_mode = False
                else:
                    self.layers_show_back = not self.layers_show_back
                if self.layers_show_back:
                    if not self.back_enabled:
                        self.back_enabled = True
                        self.back_stale = True
            elif tag == 'back_mode':
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_stale = True
                else:
                    if self.back_mode == 'grad':
                        self.back_mode = 'deconv'
                        self.back_stale = True
                    else:
                        self.back_enabled = False
            elif tag == 'back_filt_mode':
                    if self.back_filt_mode == 'raw':
                        self.back_filt_mode = 'gray'
                    elif self.back_filt_mode == 'gray':
                        self.back_filt_mode = 'norm'
                    elif self.back_filt_mode == 'norm':
                        self.back_filt_mode = 'normblur'
                    else:
                        self.back_filt_mode = 'raw'
            elif tag == 'ez_back_mode_loop':
                # Cycle:
                # off -> grad (raw) -> grad(gray) -> grad(norm) -> grad(normblur) -> deconv
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                elif self.back_mode == 'grad' and self.back_filt_mode == 'raw':
                    self.back_filt_mode = 'norm'
                elif self.back_mode == 'grad' and self.back_filt_mode == 'norm':
                    self.back_mode = 'deconv'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                else:
                    self.back_enabled = False
            elif tag == 'freeze_back_unit':
                # Freeze selected layer/unit as backprop unit
                self.backprop_selection_frozen = not self.backprop_selection_frozen
                if self.backprop_selection_frozen:
                    # Grap layer/selected_unit upon transition from non-frozen -> frozen
                    self.backprop_layer = self.layer
                    self.backprop_unit = self.selected_unit
            elif tag == 'zoom_mode':
                self.layers_pane_zoom_mode = (self.layers_pane_zoom_mode + 1) % 3
                if self.layers_pane_zoom_mode == 2 and not self.back_enabled:
                    # Skip zoom into backprop pane when backprop is off
                    self.layers_pane_zoom_mode = 0

            elif tag == 'toggle_label_predictions':
                self.show_label_predictions = not self.show_label_predictions

            elif tag == 'toggle_unit_jpgs':
                self.show_unit_jpgs = not self.show_unit_jpgs

            elif tag == 'toggle_backprop_filter':
                self.backprop_filter = not self.backprop_filter

            elif tag == 'save_descriptor':
                self.save_descriptor = True

            elif tag == 'match_descriptor':
                self.match_descriptor = not self.match_descriptor

            elif tag == 'next_descriptor':
                self.next_descriptor = True

            elif tag == 'compare_descriptor':
                self.compare_descriptor = True

            else:
                key_handled = False

            if not self.backprop_selection_frozen:
                # If backprop_selection is not frozen, backprop layer/unit follows selected unit
                if not (self.backprop_layer == self.layer and self.backprop_unit == self.selected_unit):
                    self.backprop_layer = self.layer
                    self.backprop_unit = self.selected_unit
                    self.back_stale = True    # If there is any change, back diffs are now stale

            self.drawing_stale = key_handled   # Request redraw any time we handled the key

        return (None if key_handled else key)

    def redraw_needed(self):
        with self.lock:
            return self.drawing_stale

    def move_selection(self, direction, dist = 1):
        hh,ww = self.tiles_height_width
        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += ww * dist
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= ww * dist
                if self.selected_unit < 0:
                    self.selected_unit += ww
                    self.cursor_area = 'top'
        self._ensure_valid_selected()

    def update_tiles_height_width(self, height_width, n_valid):
        '''Update the height x width of the tiles currently
        displayed. Ensures (a) that a valid tile is selected and (b)
        that up/down/left/right motion works as expected. n_valid may
        be less than prod(height_width).
        '''

        assert len(height_width) == 2, 'give as (hh,ww) tuple'
        self.tiles_height_width = height_width
        self.tiles_number = n_valid
        # If the number of tiles has shrunk, the selection may now be invalid
        self._ensure_valid_selected()

    def _ensure_valid_selected(self):
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(self.tiles_number-1, self.selected_unit)
