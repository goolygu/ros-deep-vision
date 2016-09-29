/*
 * point_cloud_manager.h
 *
 *  Created on: Jan 25, 2016
 *      Author: lku
 */

#ifndef SOURCE_DIRECTORY__UMASS_PERCEPTION_UMASS_VISION_INCLUDE_UMASS_VISION_POINT_CLOUD_MANAGER_H_
#define SOURCE_DIRECTORY__UMASS_PERCEPTION_UMASS_VISION_INCLUDE_UMASS_VISION_POINT_CLOUD_MANAGER_H_


#include "ros/ros.h"
#include <ros/package.h>
#include <image_transport/image_transport.h>

//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <cv_bridge/cv_bridge.h>
//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.
#include <sensor_msgs/image_encodings.h>
//Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
//Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/centroid.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros_deep_vision/SaveData.h>
//#include <perception_msgs/String2.h>
//#include <perception_msgs/GetPoint.h>

#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/organized.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <boost/regex.hpp>
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;
using namespace message_filters;
namespace enc = sensor_msgs::image_encodings;

class PointCloudManager
{

public:
  PointCloudManager(string image_topic, string depth_topic, string point_cloud_topic, string dataset, bool mm_depth);
  virtual
  ~PointCloudManager();
  bool create_mask_file_from_pcd(string file_name, string path);
  bool debug_multi(string file_name);

private:
  ros::NodeHandle node_;
  void vector_to_image(vector<int> mask, cv::Mat& mask_image);
  void dilation(cv::Mat& mask);
  void remove_boarder(cv::Mat& mask);
  void segment_table(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr);
  void segment_table(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, vector<int> mask);
  void get_below_table_mask(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, vector<int>& mask);
  void save_cluster_box(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, string name);
  void save_large_cluster_box(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, string name);

  void callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth, const sensor_msgs::PointCloud2ConstPtr& point_cloud);
  bool handle_save_data(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res);
  bool handle_save_data_current(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res);
  bool handle_save_data_multi(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res);
//  bool handle_show_cloud(perception_msgs::String2::Request &req, perception_msgs::String2::Response &res);
//  bool handle_get_centroid(perception_msgs::GetPoint::Request &req, perception_msgs::GetPoint::Response &res);
  bool visualize(Eigen::Vector4f point);
//  void depth_callback(const sensor_msgs::ImageConstPtr& depth);
//  void point_cloud_callback(const sensor_msgs::PointCloud2Ptr& point_cloud);
  int mm_depth_;
  int dilation_elem;
  cv_bridge::CvImagePtr image_ptr_;
  cv_bridge::CvImagePtr depth_ptr_;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr_;
  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
  Synchronizer<MySyncPolicy> *sync_;
  message_filters::Subscriber<sensor_msgs::Image> *image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> *point_cloud_sub_;
  ros::Publisher mask_pub_;
  ros::Publisher cloud_pub_;
  ros::Publisher point_pub_;
  ros::ServiceServer save_data_service_;
  ros::ServiceServer save_data_service_current_;
  ros::ServiceServer save_data_service_multi_;
  ros::ServiceServer show_cloud_service_;
  ros::ServiceServer get_centroid_service_;
  string ros_path_;
  string data_folder_;
  string current_folder_;
};



#endif /* SOURCE_DIRECTORY__UMASS_PERCEPTION_UMASS_VISION_INCLUDE_UMASS_VISION_POINT_CLOUD_MANAGER_H_ */
