# ROS Deep Vision package

This is a ROS package that generates hierarchical CNN features and their locations based on RGB-D inputs. Hierarchcial CNN features represent meaningful properties of object parts and can be localized to support manipulation. Read the arXiv paper [Associating Grasping with Convolutional Neural Network Features](https://arxiv.org/abs/1609.03947) for more details. This repository is originally forked from the Deep Visualization Toolbox made by Yosinski which I found extremely useful in understanding CNNs. 

## Assumptions

This current version assumes objects are placed on the ground or table top in order to crop the image into square images centered on objects as CNN inputs. Only the top N largest clusters are handled. This package is setup to handle RGB-D camera inputs with resolution 640x480 such as the Kinect and Asus xtion.

## Installation

This package requires a specific branch of caffe and several different libraries listed below. This guide also assumes ROS (Version >= hydro) is already installed.

### Step 0: Install caffe

Get the master branch of [caffe](http://caffe.berkeleyvision.org/) to compile on your machine. If you've never used Caffe before, it can take a bit of time to get all the required libraries in place. Fortunately, the [installation process is well documented](http://caffe.berkeleyvision.org/installation.html).

In addition to running on GPU with CUDA it is highly recommended to install the cudnn library to speed up the computation. 
Remember to set USE_CUDNN := 1 in Caffe's Makefile.config before compiling if cudnn is installed.

After installing CUDA and caffe make sure that the following environment variables are set correctly:

export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD\_LIBRARY\_PATH=/usr/local/cuda-7.0/lib64:$LD\_LIBRARY\_PATH
export PYTHONPATH=$PYTHONPATH:{PATH}/caffe/python

### Step 1: Compile the ros branch of caffe

Instead of using the master branch of caffe, to use the package
you'll need a slightly modified branch. Getting the branch and switching to it is easy.

If you are using cudnn version 5 than starting from your caffe directory, run:

    $ git remote add goolygu https://github.com/goolygu/caffe.git
    $ git fetch --all
    $ git checkout --track -b ros-cudnn5 goolygu/ros-cudnn5
    $ make clean
    $ make -j
    $ make -j pycaffe

If you are using cudnn version 4 than starting from your caffe directory, run:

    $ git remote add goolygu https://github.com/goolygu/caffe.git
    $ git fetch --all
    $ git checkout --track -b ros goolygu/ros
    $ make clean
    $ make -j
    $ make -j pycaffe

If you are not using cudnn, both versions should work.


### Step 2: Install required python libraries if haven't

    $ sudo apt-get install python-opencv

Install [pip](https://pip.pypa.io/en/stable/installing/) if haven't.

    $ sudo pip install scipy
    $ sudo pip install scikit-learn
    $ sudo pip install scikit-image

Download [python-pcl](https://github.com/strawlab/python-pcl) and install from local directory.
    
    $ sudo pip install -e ./python-pcl/

### Step 3: Download and configure ros-deep-vision package

Download the package and place it along with other ROS packages in the catkin workspace.

    $ git clone https://github.com/goolygu/ros-deep-vision
    $ roscd ros_deep_vision

modify `./src/settings.py` so the `caffevis_caffe_root` variable points to the directory where you've compiled caffe in Step 1:

Download the example model weights and corresponding top-9 visualizations made by Yosinski (downloads a 230MB model and 1.1GB of jpgs to show as visualization):

    $ cd models/caffenet-yos/
    $ ./fetch.sh

### Step 5: Install required ros packages if haven't

    $ sudo apt-get install ros-{rosversion}-openni2-launch

I would recommend modifying the depth registration option in "openni2_launch/launch/openni2.launch" if the point cloud color has an offset and your hardware supports depth registration.
<arg name="depth_registration" default="true" />

### Step 4: Run the package

Make sure your RGB-D camera is connected.
Start the RGB-D camera

    $ roslaunch openni2_launch openni2.launch

Start the input server that does point cloud segmentation 

    $ roslaunch ros_deep_vision input_server.launch

Start rviz

    $ roslaunch ros_deep_vision rviz.launch

You should be able to see the point cloud in rviz.
Start the cnn state manager that generates the features

    $ roslaunch ros_deep_vision cnn_state_manager.launch

Press enter r in the cnn\_state_manager to run.
The detected features should show up in rviz similar to the following image when finished running. The yellow, cyan, and magenta dots represent conv-5, conv-4, and conv-3 hierarchical CNN features. 
Set ```self.max_clusters = 3``` in ```cnn_state_manager.py``` to a higher number if more than 3 objects are in the scene.
![alt tag](https://github.com/goolygu/ros-deep-vision/blob/master/doc/rviz_feature_visualization.png?raw=true)

Set ```self.data_monster.show_backprop = True``` in ```cnn_state_manager.py``` if you want to visualize the targeted backpropagation result for each hierarchical CNN feature like in the image below. The blue dots are the feature locations based on the average response locations. Note that this may generate a bunch of image windows, set ```self.max_clusters = 1``` so that it only handles the largest object.  You can also change the following settings: 
```python
elif case == "cnn_features":
    self.conv5_top = 10
    self.conv4_top = 5
    self.conv3_top = 2
    self.conv2_top = 0
```
in ```data_settings.py``` to modify the number of features extracted. Note that in this case there will be 10 conv5, 50 conv4, and 100 conv3 hierarchical CNN features.

![alt tag](https://github.com/goolygu/ros-deep-vision/blob/master/doc/backprop.png?raw=true)

