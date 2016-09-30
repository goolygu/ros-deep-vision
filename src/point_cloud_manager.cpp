/*
 * point_cloud_manager.cpp
 *
 *  Created on: Jan 25, 2016
 *      Author: lku
 */
#include <ros_deep_vision/point_cloud_manager.h>


boost::mutex mask_point_cloud_mutex;

PointCloudManager::PointCloudManager(string image_topic, string depth_topic, string point_cloud_topic, string dataset, bool mm_depth)
{
  ros_path_ = ros::package::getPath("ros_deep_vision");
  data_folder_ = "/data/" + dataset + "/";
  current_folder_ = "/current/";
  image_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(node_, image_topic, 1);
  depth_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(node_, depth_topic, 1);
  point_cloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, point_cloud_topic, 1);

  mask_pub_ =  node_.advertise<sensor_msgs::Image>("/image_mask", 1);
  cloud_pub_ =  node_.advertise<sensor_msgs::PointCloud2>("/point_cloud", 1);

  cloud_ptr_ = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
  mm_depth_ = mm_depth;
  dilation_elem = 2;

  sync_ = new Synchronizer<MySyncPolicy>(MySyncPolicy(10), *image_sub_, *depth_sub_, *point_cloud_sub_);
  sync_->registerCallback(boost::bind(&PointCloudManager::callback, this, _1, _2, _3));

  cout << "synchronizer ready" << endl;
  save_data_service_ = node_.advertiseService("save_point_cloud", &PointCloudManager::handle_save_data, this);
  save_data_service_current_ = node_.advertiseService("save_point_cloud_current", &PointCloudManager::handle_save_data_current, this);
  save_data_service_multi_ = node_.advertiseService("save_point_cloud_multi", &PointCloudManager::handle_save_data_multi, this);
  show_cloud_service_ = node_.advertiseService("show_point_cloud", &PointCloudManager::handle_show_cloud, this);
//  get_centroid_service_ = node_.advertiseService("get_centroid", &PointCloudManager::handle_get_centroid, this);
  point_pub_ =  node_.advertise<visualization_msgs::Marker>("/point_cloud_center", 10);
}

PointCloudManager::~PointCloudManager() {

}

bool PointCloudManager::visualize(Eigen::Vector4f point)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "/r2/head/asus_depth_optical_frame";//"r2/robot_reference"; //"r2/simulated_asus_depth_frame";//
  marker.header.stamp = ros::Time::now();
  marker.ns = "sift";
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = point[0];
  marker.pose.position.y = point[1];
  marker.pose.position.z = point[2];
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  // Set the scale of the marker -- length, diameter
  marker.scale.x = 0.02;
  marker.scale.y = 0.02;
  marker.scale.z = 0.02;
  // Set the color -- be sure to set alpha to something non-zero!
  marker.color.r = 1.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;
  point_pub_.publish(marker);

  return true;
}

void PointCloudManager::callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth, const sensor_msgs::PointCloud2ConstPtr& point_cloud)
{
  try
  {
      //OpenCV expects color images to use BGR channel order.

//        boost::mutex::scoped_lock lock(sift_depth_mutex);
//      cout << "got callback" << endl;
      cout << ".";
      cout.flush();
      image_ptr_ = cv_bridge::toCvCopy(image, enc::BGR8);
      depth_ptr_ = cv_bridge::toCvCopy(depth, enc::TYPE_32FC1);
      cv_bridge::CvImagePtr mask_ptr_ = cv_bridge::toCvCopy(depth, enc::TYPE_32FC1);
      if (mm_depth_)
      {
        depth_ptr_->image = depth_ptr_->image / float(1000);
      }
      {
      boost::mutex::scoped_lock lock(mask_point_cloud_mutex);
      pcl::fromROSMsg(*point_cloud, *cloud_ptr_);

      vector<int> mask(cloud_ptr_->points.size(),0);
      cv::Mat mask_image(depth_ptr_->image);
      this->get_below_table_mask(cloud_ptr_, mask);

      this->vector_to_image(mask, mask_image);
      this->remove_boarder(mask_image);
      this->dilation(mask_image);
      mask_ptr_->image = mask_image;

      mask_pub_.publish(mask_ptr_->toImageMsg());
      }

  }
  catch (cv_bridge::Exception& e)
  {
      //if there is an error during conversion, display it
      ROS_ERROR("mask creator exception: %s", e.what());
      return;
  }
}

void PointCloudManager::vector_to_image(vector<int> mask, cv::Mat& mask_image)
{
//  cout << "mask size " << mask.size() << " image size " << mask_image.cols << " r " << mask_image.rows << endl;

  for (int i = 0; i < mask.size(); i++)
  {

    int r = round(i / mask_image.cols);
    int c = round(i % mask_image.cols);

    if (mask.at(i) == -1) {
      mask_image.at<float>(r,c) = float(0);
    }
    else {
      mask_image.at<float>(r,c) = float(255);
    }
  }
}


void PointCloudManager::get_below_table_mask(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, vector<int>& mask)
{

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
  //cout << "segment" << endl;
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud(cloud_ptr);
  seg.segment (*inliers, *coefficients);

//  cout << "cloud ptr size " << cloud_ptr->points.size() << endl;

//  cout << "inlier size " << inliers->indices.size() << endl;
  for (int i = 0; i < inliers->indices.size(); i++)
  {
    mask[inliers->indices[i]] = -1;
  }

  // remove anything below the table
  for(size_t i = 0; i< cloud_ptr->points.size(); i++)
  {
    double d = pcl::pointToPlaneDistanceSigned (cloud_ptr->points[i], coefficients->values[0],coefficients->values[1], coefficients->values[2], coefficients->values[3]);
    if(d > -0.02)
    {
      mask[i] = -1;
    }

  }

//  pcl::ExtractIndices<pcl::PointXYZRGBA> extract ;
//  extract.setInputCloud (cloud_ptr);
//  extract.setIndices (inliers);
//  extract.setNegative (true);
//  extract.filter (*cloud_ptr);


//  extract.setNegative (false);
//
//  extract.filter (*cloud_table_ptr);

//  visualize_plane(cloud_table_ptr, coefficients);

}

void PointCloudManager::segment_table(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr)
{
  vector<int> mask(cloud_ptr->points.size(),0);
  get_below_table_mask(cloud_ptr, mask);
  segment_table(cloud_ptr, mask);
}

void PointCloudManager::segment_table(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, vector<int> mask)
{
//  cout << "cloud size before " << cloud_ptr->size() << endl;
//  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
//  //cout << "segment" << endl;
//  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
//  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
//  pcl::PointIndices::Ptr outliers (new pcl::PointIndices);
//  // Create the segmentation object
//  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
//  // Optional
//  seg.setOptimizeCoefficients (true);
//  // Mandatory
//  seg.setModelType (pcl::SACMODEL_PLANE);
//  seg.setMethodType (pcl::SAC_RANSAC);
//  seg.setDistanceThreshold (0.01);
//
//  seg.setInputCloud(cloud_ptr);
//  seg.segment (*inliers, *coefficients);
//
//  cout << "cloud ptr size " << cloud_ptr->points.size() << endl;
  const float bad_point = std::numeric_limits<float>::quiet_NaN();
//


  int count = 0;
  for (int i=0; i < mask.size(); i++)
  {
    if (mask[i] == -1)
    {
      cloud_ptr->points[i].x = bad_point;
      cloud_ptr->points[i].y = bad_point;
      cloud_ptr->points[i].z = bad_point;
      count++;
    }
  }
  cout << "inlier size " << mask.size() << endl;

//  for (int i = 0; i < inliers->indices.size(); i++)
//  {
//    int idx = inliers->indices[i];
//    cloud_ptr->points[idx].x = bad_point;
//    cloud_ptr->points[idx].y = bad_point;
//    cloud_ptr->points[idx].z = bad_point;
//  }
//
//  // remove anything below the table
//  for(size_t i = 0; i< cloud_ptr->points.size(); i++)
//  {
//    double d = pcl::pointToPlaneDistanceSigned (cloud_ptr->points[i], coefficients->values[0],coefficients->values[1], coefficients->values[2], coefficients->values[3]);
//    if(d > -0.02)
//    {
//      cloud_ptr->points[i].x = bad_point;
//      cloud_ptr->points[i].y = bad_point;
//      cloud_ptr->points[i].z = bad_point;
//    }
//
//  }

//
//  cout << "cloud size " << cloud_ptr->size() << endl;
//


}

void PointCloudManager::save_cluster_box(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, string name)
{
   vector<int> mapping;

   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_clean (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr));
   cloud_clean->is_dense = false;
   pcl::removeNaNFromPointCloud(*cloud_clean, *cloud_clean, mapping);

   pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
//   pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>);

   tree->setInputCloud (cloud_clean);

   std::vector<pcl::PointIndices> cluster_indices;
   pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
   ec.setClusterTolerance (0.02); // 2cm
   ec.setMinClusterSize (100);
   ec.setMaxClusterSize (2500000);

   ec.setSearchMethod (tree);

   ec.setInputCloud (cloud_clean);

   ec.extract (cluster_indices);

   ostringstream box_xy_ss;

   int j = 0;
   for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
   {
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
     double sum_x = 0, sum_y = 0;
     int num_points = 0;
     int min_x=10000, min_y=10000, max_x=0, max_y=0;
     for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
     {
       int idx_orig = mapping[*pit];
       int x = idx_orig / cloud_ptr->width;
       int y = idx_orig % cloud_ptr->width;
       sum_x += x;
       sum_y += y;
       if (x < min_x)
         min_x = x;
       if (x > max_x)
         max_x = x;
       if (y < min_y)
         min_y = y;
       if (y > max_y)
         max_y = y;
       num_points++;
       cloud_cluster->points.push_back (cloud_ptr->points[*pit]); //*
     }

     double avg_x = sum_x/num_points;
     double avg_y = sum_y/num_points;

     box_xy_ss << min_x << "," << max_x << "," << min_y << "," << max_y << endl;

     cout << box_xy_ss.str();

     cloud_cluster->width = cloud_cluster->points.size ();
     cloud_cluster->height = 1;
     cloud_cluster->is_dense = true;

     std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
     std::stringstream ss;
     ss << "cloud_cluster_" << j << ".pcd";
//     writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
     j++;
   }

   ofstream file;
   file.open(name.append("_box.txt").c_str());
   file << box_xy_ss.str();
   file.close();

}

void PointCloudManager::save_large_cluster_box(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, string name)
{
   vector<int> mapping;

   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_clean (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr));
   cloud_clean->is_dense = false;
   pcl::removeNaNFromPointCloud(*cloud_clean, *cloud_clean, mapping);

   pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
//   pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::OrganizedNeighbor<pcl::PointXYZRGBA>);

   tree->setInputCloud (cloud_clean);

   std::vector<pcl::PointIndices> cluster_indices;
   pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
   ec.setClusterTolerance (0.04); // 4cm
   ec.setMinClusterSize (1000);
   ec.setMaxClusterSize (2500000);

   ec.setSearchMethod (tree);

   ec.setInputCloud (cloud_clean);

   ec.extract (cluster_indices);

   ostringstream box_xy_ss;

   int min_x=10000, min_y=10000, max_x=0, max_y=0;

   for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
   {
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);

     for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
     {
       int idx_orig = mapping[*pit];
       int x = idx_orig / cloud_ptr->width;
       int y = idx_orig % cloud_ptr->width;

       if (x < min_x)
         min_x = x;
       if (x > max_x)
         max_x = x;
       if (y < min_y)
         min_y = y;
       if (y > max_y)
         max_y = y;
       cloud_cluster->points.push_back (cloud_ptr->points[*pit]); //*
     }




     cloud_cluster->width = cloud_cluster->points.size ();
     cloud_cluster->height = 1;
     cloud_cluster->is_dense = true;

     std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;

   }
   box_xy_ss << min_x << "," << max_x << "," << min_y << "," << max_y << endl;
   cout << box_xy_ss.str();
   ofstream file;
   file.open(name.append("_box.txt").c_str());
   file << box_xy_ss.str();
   file.close();

}

void PointCloudManager::remove_boarder(cv::Mat& mask)
{
  int margin_c = 50;
  int margin_r = 30;
  for (int i=0; i < mask.cols; i++)
  {
    for (int j=0; j < mask.rows; j++)
    {
      if (i < margin_c || i > mask.cols - margin_c || j < margin_r || j > mask.rows - margin_r)
      {
        mask.at<float>(j,i) = float(0);
      }
    }
  }
}

void PointCloudManager::dilation(cv::Mat& mask)
{
  int dilation_type;
  int dilation_size = 5;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  Mat dilated_mask;
  dilate( mask, dilated_mask, element );
  mask = dilated_mask;
}

bool PointCloudManager::handle_save_data(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res)
{

  {
  boost::mutex::scoped_lock lock(mask_point_cloud_mutex);
  pcl::io::savePCDFileASCII (ros_path_ + data_folder_ + req.name + ".pcd", *cloud_ptr_);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr_seg (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr_));
  this->segment_table(cloud_ptr_seg);
  pcl::io::savePCDFileASCII (ros_path_ + data_folder_ + req.name + "_seg.pcd", *cloud_ptr_seg);
  }
  res.result = 0;

  return true;
}

bool PointCloudManager::handle_save_data_current(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res)
{

  {
  boost::mutex::scoped_lock lock(mask_point_cloud_mutex);
//  pcl::io::savePCDFileASCII (ros_path_ + "/data/current/" + req.name + ".pcd", *cloud_ptr_);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr_seg (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr_));
  pcl::io::savePCDFileASCII (ros_path_ + current_folder_ + req.name + ".pcd", *cloud_ptr_);
  this->segment_table(cloud_ptr_seg);
  pcl::io::savePCDFileASCII (ros_path_ + current_folder_ + req.name + "_seg.pcd", *cloud_ptr_seg);
  }
  res.result = 0;

  return true;
}

bool PointCloudManager::handle_save_data_multi(ros_deep_vision::SaveData::Request &req, ros_deep_vision::SaveData::Response &res)
{

  {
  boost::mutex::scoped_lock lock(mask_point_cloud_mutex);
//  pcl::io::savePCDFileASCII (ros_path_ + "/data/current/" + req.name + ".pcd", *cloud_ptr_);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr_seg (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr_));
  pcl::io::savePCDFileASCII (ros_path_ + current_folder_ + req.name + ".pcd", *cloud_ptr_);
  this->segment_table(cloud_ptr_seg);
  this->save_large_cluster_box(cloud_ptr_seg, ros_path_ + current_folder_ + req.name);

  pcl::io::savePCDFileASCII (ros_path_ + current_folder_ + req.name + "_seg.pcd", *cloud_ptr_seg);
  }
  res.result = 0;

  return true;
}

bool PointCloudManager::handle_show_cloud(ros_deep_vision::String2::Request &req, ros_deep_vision::String2::Response &res)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr;
  cloud_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
  cout << "show cloud " << req.name << endl;
  pcl::io::loadPCDFile(ros_path_ + data_folder_ +req.name+ ".pcd", *cloud_ptr);
  sensor_msgs::PointCloud2 point_cloud ;

  pcl::toROSMsg(*cloud_ptr, point_cloud);
  point_cloud.header.frame_id = "/r2/head/asus_depth_optical_frame";//simulated_asus_frame";
  cloud_pub_.publish(point_cloud);
  res.result = 0;

  return true;
}

/*
bool PointCloudManager::handle_get_centroid(perception_msgs::GetPoint::Request &req, perception_msgs::GetPoint::Response &res)
{
  {
  boost::mutex::scoped_lock lock(mask_point_cloud_mutex);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr_seg (new pcl::PointCloud<pcl::PointXYZRGBA>(*cloud_ptr_));
  this->segment_table(cloud_ptr_seg);
  Eigen::Vector4f centroid;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud_ptr_clean;
  for (int i = 0; i < cloud_ptr_seg->size(); i++)
  {
    if (pcl_isfinite(cloud_ptr_seg->points[i].x)) {
      cloud_ptr_clean.points.push_back(cloud_ptr_seg->points[i]);
    }
  }

  unsigned int count = pcl::compute3DCentroid(cloud_ptr_clean, centroid);
  cout << "count " << count << endl;
  cout << "centroid " << centroid << endl;
  res.position.x = centroid[0];
  res.position.y = centroid[1];
  res.position.z = centroid[2];
  visualize(centroid);
  }

  return true;
}
*/
bool PointCloudManager::create_mask_file_from_pcd(string file_name, string path)
{

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBA>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (path + "/" + file_name + ".pcd", *cloud_ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file\n");
    return false;
  }
  cout << file_name << endl;
  vector<int> mask(cloud_ptr->points.size(),0);
//  cv::Mat mask_image = cv::Mat::zeros(640, 480, CV_8UC1);
  this->segment_table(cloud_ptr);
  pcl::io::savePCDFileASCII (path + "/" + file_name + "_seg.pcd", *cloud_ptr);
//  this->vector_to_image(mask, mask_image);

//  imwrite(path+file_name+"_masko.png", mask_image);
  return true;
}

bool PointCloudManager::debug_multi(string file_name)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBA>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (ros_path_ + current_folder_ + file_name + ".pcd", *cloud_ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file\n");
    return false;
  }
  cout << file_name << endl;
  cv::Mat mask_image(480,640,CV_32F);

  vector<int> mask(cloud_ptr->points.size(),0);
  get_below_table_mask(cloud_ptr, mask);
  segment_table(cloud_ptr, mask);

  this->vector_to_image(mask, mask_image);
  this->remove_boarder(mask_image);
  this->dilation(mask_image);

  cv::imshow("output", mask_image);
  cv::waitKey(0);

  this->save_cluster_box(cloud_ptr, ros_path_ + current_folder_ + file_name + "_debug");

}




