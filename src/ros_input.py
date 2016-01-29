#!/usr/bin/env python
import roslib
roslib.load_manifest('ros_deep_vision')
import sys
import rospy
import cv2
import threading
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from image_misc import convert_depth_image_to_rgb


class ImageConverter:

  def __init__(self):

    rospy.init_node('image_converter', anonymous=True)

    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    # cv2.namedWindow("Image window", 1)
    self.depth_input = False
    self.use_mask = True
    self.latest_frame = None;
    self.latest_depth = None;
    self.latest_mask = None;

    self.bridge = CvBridge()
    # self.image_sub = rospy.Subscriber("/asus/rgb/image_raw",Image,self.callback,queue_size=1)
    self.image_sub = rospy.Subscriber("/r2/head/asus/rgb/image_raw",Image,self.callback,queue_size=1)
    self.depth_sub = rospy.Subscriber("/r2/head/asus/depth/image_raw",Image,self.depth_callback,queue_size=1)
    self.mask_sub = rospy.Subscriber("/image_mask",Image, self.mask_callback, queue_size=1)

    self.lock = threading.Lock()

  def get_frame(self):
      if self.depth_input:
          if not self.latest_depth is None:
              with self.lock:
                  depth_rgb = convert_depth_image_to_rgb(self.latest_depth)
                  return depth_rgb
          else:
              return None
      else:
          if not self.latest_frame is None:
              with self.lock:
                  latest_frame_rgb = self.latest_frame[:,:,::-1]
                  return latest_frame_rgb
          else:
              return None


  def get_mask(self):

      if not self.use_mask:
          if self.latest_frame is None:
              return None
          else:
              empty_mask = np.zeros([self.latest_frame.shape[0],self.latest_frame.shape[1],1])
              empty_mask.fill(255)
              return empty_mask

      if not self.latest_mask is None:
          with self.lock:
            return self.latest_mask

      else:
          return None

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      with self.lock:
          self.latest_frame = cv_image
    except CvBridgeError, e:
      print e

  def depth_callback(self,data):
      try:
        cv_depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        with self.lock:
            self.latest_depth = cv_depth
      except CvBridgeError, e:
        print e



  def mask_callback(self,data):
    try:
      mask_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
      with self.lock:
          self.latest_mask = mask_image
    except CvBridgeError, e:
      print e

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError, e:
    #   print e

def main(args):
  ic = ImageConverter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
