# Settings for live_vis

import os
root_dir = os.path.dirname(os.path.abspath(__file__))
ros_dir = os.path.dirname(os.path.dirname(root_dir))

# Set this to point to your compiled checkout of caffe
caffevis_caffe_root      = '/home/lku/Workspace/caffe'

if not os.path.exists(caffevis_caffe_root):
    raise Exception('ERROR: Set caffevis_caffe_root in settings.py first.')

# Global settings
input_updater_capture_device = 0              # 0 default, on Mac works for builtin camera or external USB webcam
input_updater_sleep_after_read_frame = 1.0 #1.0/20    # Sleep after reading frame
#input_updater_sleep_after_read_frame = 2    # Sleep after reading frame
input_updater_heartbeat_required = 15.0      # Thread dies after this many seconds without a heartbeat
main_loop_sleep_ms = 1                    # Sleep while waiting for key presses and redraws. Recommendation: 1

window_panes = (
    # (i, j, i_size, j_size)
    ('input',            (  0,    0,  300,   300)),    # Probably include this
    ('caffevis_aux',     (300,    0,  300,   300)),
    ('caffevis_back',    (600,    0,  300,   300)),
    ('caffevis_status',  (900,    0,   30,  1500)),
    ('caffevis_control', (  0,  300,   30,   870)),
    ('caffevis_layers',  ( 30,  300,  870,   870)),
    ('caffevis_jpgvis',  (  0, 1175,  900,   300)),
)
# Define global_scale as a float to rescale window and all panes. Handy for quickly changing resolution for a different screen.
global_scale = None
global_scale = .9
# Define global_font_size to scale all font sizes by this amount.
global_font_size = .9

if global_scale is not None:
    scaled_window_panes = []
    for wp in window_panes:
        scaled_window_panes.append([wp[0], [int(val*global_scale) for val in wp[1]]])
    window_panes = scaled_window_panes
help_pane_loc = (.07, .07, .86, .86)    # as a fraction of main window
window_background = (.2, .2, .2)
#window_background = (0, 0, 0)
stale_background = (.3, .3, .2)
#stale_background = (0, 0, 1)
static_files_dir = ros_dir + '/input_images'
static_files_regexp = '.*\.(jpg|jpeg|png)$'
static_files_ignore_case = True
static_file_stretch_mode = False        # True to stretch to square, False to crop to square. (Can change at runtime via 'stretch_mode' key.)


keypress_pause_handle_iterations = 2    # int, 0+. How many times to go through the main loop after a keypress before resuming handling frames (0 to handle every frame as it arrives). Setting this to a value > 0 can enable more responsive keyboard input even when other settings are tuned to maximize the framerate. Default: 2
keypress_pause_redraw_iterations = 1    # int, 0+. How many times to go through the main loop after a keypress before resuming redraws (0 to redraw every time it is needed). Setting this to a value > 0 can enable more responsive keyboard input even when other settings are tuned to maximize the framerate. Default: 1
redraw_at_least_every = 3               # int, 1+. Force a redraw even when keys are pressed if there have been this many passes through the main loop without a redraw due to the keypress_pause_redraw_iterations setting combined with many key presses. Default: 3.

installed_apps = (
    ('caffevis.app', 'CaffeVisApp'),
)


# Settings for caffevis
caffevis_heartbeat_required = 15.0      # Thread dies after this many seconds without a heartbeat

caffevis_help_face = 'FONT_HERSHEY_COMPLEX_SMALL'
caffevis_help_loc = (20,10)   # r,c order
caffevis_help_line_spacing = 10     # extra pixel spacing between lines
caffevis_help_clr   = (1,1,1)
caffevis_help_fsize = 1.0 * global_font_size
caffevis_help_thick = 1


caffevis_deploy_prototxt = ros_dir + '/models/caffenet-yos/caffenet-yos-deploy.prototxt'
caffevis_network_weights = ros_dir + '/models/caffenet-yos/caffenet-yos-weights'
caffevis_data_mean       = ros_dir + '/models/caffenet-yos/ilsvrc_2012_mean.npy'
caffevis_labels          = ros_dir + '/models/caffenet-yos/ilsvrc_2012_labels.txt'
caffevis_unit_jpg_dir    = ros_dir + '/models/caffenet-yos/unit_jpg_vis'

caffevis_data_hw         = (227,227)
caffevis_label_layers    = ('fc8', 'prob')
caffevis_mode_gpu        = True
caffevis_pause_after_keys = .10     # Pause Caffe forward/backward computation for this many seconds after a keypress. This is to keep the processor free for a brief period after a keypress, which allow the interface to feel much more responsive. After this period has passed, Caffe resumes computation, in CPU mode often occupying all cores. Default: .1
caffevis_frame_wait_sleep = .01
#caffevis_frame_wait_sleep = 2.0
caffevis_jpg_load_sleep = .01
#caffevis_jpg_load_sleep = 2.0
caffevis_fast_move_dist = 3      # How far to move when using fast left/right/up/down keys
# Size of jpg reading cache in bytes (default: 2GB)
# Note: largest fc6/fc7 images are ~600MB. Cache smaller than this will be painfully slow when using patterns_mode for fc6 and fc7.
# Cache use when all layers have been loaded is ~1.6GB
caffevis_jpg_cache_size  = 2000*1024**2

caffevis_grad_norm_blur_radius = 4.0

caffevis_boost_indiv_choices = (0, .3, .5, .8, 1)
caffevis_boost_indiv_default_idx = 0   # index into above list
caffevis_boost_gamma_choices = (1, .7, .5, .3)
caffevis_boost_gamma_default_idx = 0   # index into above list
caffevis_init_show_label_predictions = True
caffevis_init_show_unit_jpgs = True

caffevis_control_face = 'FONT_HERSHEY_COMPLEX_SMALL'
caffevis_control_loc = (15,0)   # r,c order
#caffevis_control_clr = [int(v*255) for v in (.8,.8,.8)]
#caffevis_control_clr_selected = [int(v*255) for v in (1, 1, 1)]
#caffevis_control_clr_cursor = [int(v*255) for v in (.5,1,.5)]
caffevis_control_clr = (.8,.8,.8)
caffevis_control_clr_selected = (1, 1, 1)
caffevis_control_clr_cursor = (.5,1,.5)
caffevis_control_clr_bp = (.8, .8, 1)
caffevis_control_fsize = 1.0 * global_font_size
caffevis_control_thick = 1
caffevis_control_thick_selected = 2
caffevis_control_thick_cursor = 2
caffevis_control_thick_bp = 2

#caffevis_layer_clr_cursor = [int(v*255) for v in (.5,1,.5)]
caffevis_layer_clr_cursor   = (.5,1,.5)
caffevis_layer_clr_back_background = (.2,.2,.5)
caffevis_layer_clr_back_sel = (.2,.2,1)

caffevis_status_face = 'FONT_HERSHEY_COMPLEX_SMALL'
caffevis_status_loc = (20,10)   # r,c order
caffevis_status_line_spacing = 5     # extra pixel spacing between lines
#caffevis_status_clr = [int(v*255) for v in (.8,.8,.8)]
caffevis_status_clr = (.8,.8,.8)
#caffevis_status_clr_selected = (.5,1,.5)
caffevis_status_fsize = 1.0 * global_font_size
caffevis_status_thick = 1
#caffevis_status_thick_selected = 2
caffevis_jpgvis_stack_vert = True

caffevis_class_face = 'FONT_HERSHEY_COMPLEX_SMALL'
caffevis_class_loc = (20,10)   # r,c order
caffevis_class_line_spacing = 10     # extra pixel spacing between lines
caffevis_class_clr_0 = (.5,.5,.5)
caffevis_class_clr_1 = (.5,1,.5)
caffevis_class_fsize = 1.0 * global_font_size
caffevis_class_thick = 1
