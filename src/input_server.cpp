#include <ros/ros.h>
// PCL specific includes

#include "ros_deep_vision/point_cloud_manager.h"

using namespace std;


int
main (int argc, char** argv)
{
	bool debug = false;


	// Initialize ROS
	ros::init (argc, argv, "input_server");

	ros::NodeHandle nh("~");
	bool mm_depth = false;

	string mode = "sim";
	if (nh.getParam("mode", mode))
	{
		nh.deleteParam("mode");
	}
	cout << "mode: " << mode << endl;

	if (mode.compare("asus") == 0)
	{
		mm_depth = true;
	}

  string dataset = "set0";
  if (nh.getParam("dataset", dataset))
  {
    nh.deleteParam("dataset");
  }
  cout << "dataset: " << dataset << endl;

  PointCloudManager point_cloud_manager("/asus/rgb/image_raw","/asus/depth/image_raw","/asus/depth_registered/points", dataset, true);

  string file_name = "";

  if (nh.getParam("file", file_name))
  {
    nh.deleteParam("file");
    point_cloud_manager.debug_multi(file_name);
  }

  ros::spin();


}
