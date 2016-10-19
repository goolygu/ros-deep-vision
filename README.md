# ROS Deep Vision package

This is a ROS package that generates hierarchical CNN features and their locations based on RGB-D inputs. Hierarchcial CNN features represent meaningful properties of object parts and can be localized to support manipulation. Read the arXiv paper [Associating Grasping with Convolutional Neural Network Features](https://arxiv.org/abs/1609.03947) for more details. This repository is originally forked from the Deep Visualization Toolbox made by Yosinski which I found extremely useful in understanding CNNs. 

## Assumptions

This current version assumes objects are placed on the ground or table top in order to crop the image into square images centered on objects as CNN inputs.

## Installation

This package requires a specific branch of caffe and several different libraries listed below. This guide also assumes ROS (Version >= hydro) is already installed.

### Step 0: Install caffe

Get the master branch of [caffe](http://caffe.berkeleyvision.org/) to compile on your machine. If you've never used Caffe before, it can take a bit of time to get all the required libraries in place. Fortunately, the [installation process is well documented](http://caffe.berkeleyvision.org/installation.html).

In addition to running on GPU with CUDA it is highly recommended to install the cudnn library to speed up the computation. 
Remember to set USE_CUDNN := 1 in Caffe's Makefile.config before compiling if cudnn is installed.

After installing CUDA adn caffe make sure that the following environment variables are set correctly:

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
    $ roscd ros-deep-vision

modify `./src/settings.py` so the `caffevis_caffe_root` variable points to the directory where you've compiled caffe in Step 1:

Download the example model weights and corresponding top-9 visualizations made by Yosinski (downloads a 230MB model and 1.1GB of jpgs to show as visualization):

    $ cd models/caffenet-yos/
    $ ./fetch.sh

### step 5: Install required ros packages if haven't

$ sudo apt-get install ros-{rosversion}-openni2-launch

I would recommend modifying the depth registration option in "openni2_launch/launch/openni2.launch" if the point cloud color has an offset and your hardware supports.
<arg name="depth_registration" default="true" />

### Step 4: Run the package

Make sure your rgbd camera is connected.
Start the rgbd camera
$ roslaunch openni2_launch openni2.launch

Start the input server that does point cloud segmentation 
$ roslaunch ros\_deep\_vision input\_server.launch

Start rviz
$ roslaunch ros\_deep\_vision rviz.launch

Start the cnn state manager that generates the features
$ roslaunch ros\_deep\_vision cnn\_state_manager.launch

Press enter r in the cnn\_state_manager to run.
The detected features should show up in rviz when finished running.





