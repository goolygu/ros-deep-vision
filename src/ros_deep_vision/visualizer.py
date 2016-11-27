#!/usr/bin/env python
import sys
import copy
import rospy
import geometry_msgs.msg
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import tf

import numpy as np

from tf.transformations import quaternion_from_euler
from tf.transformations import *


class Visualizer:

    def __init__(self):
        self.marker_pub = {}

    def set_topics(self, topic_list):
        for topic in topic_list:
            self.marker_pub[topic] = rospy.Publisher(topic, Marker, queue_size=100)

    def set_frame(self, frame):
        self.frame = frame

    def array_to_point_list(self, array_list):
        point_list = []
        for point in array_list:
            if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
                continue
            p_msg = Point()
            p_msg.x = point[0]
            p_msg.y = point[1]
            p_msg.z = point[2]
            point_list.append(p_msg)
        return point_list

    def show_point(self, xyz, idx, ns, topic, color, alpha=1, scale=0.05):
        marker = Marker()

        marker.header.frame_id = self.frame
        marker.header.stamp = rospy.Time()
        marker.type = marker.SPHERE
        marker.id = idx
        marker.ns = ns
        marker.action = marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = alpha
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        if not topic in self.marker_pub:
            self.marker_pub[topic] = rospy.Publisher(topic, Marker, queue_size=100)
        self.marker_pub[topic].publish(pl_marker)

    def publish_point_array(self, point_array, idx, ns, topic, color, m_type=Marker.POINTS, alpha=0.5, scale=0.01):
        point_list = self.array_to_point_list(point_array)
        self.publish_point_list(point_list, idx, ns, topic, color, m_type, alpha, scale)

    def publish_point_list(self, point_list, idx, ns, topic, color, m_type=Marker.POINTS, alpha=0.5, scale=0.01):

        pl_marker = Marker()
        pl_marker.header.frame_id = self.frame
        pl_marker.header.stamp = rospy.Time()
        pl_marker.id = idx
        pl_marker.ns = ns
        pl_marker.type = m_type
        # pl_marker.pose.position.x = 1
        # pl_marker.pose.position.y = 1
        # pl_marker.pose.position.z = 1
        pl_marker.scale.x = scale
        pl_marker.scale.y = scale
        pl_marker.scale.z = scale
        pl_marker.color.a = alpha
        pl_marker.color.r = color[0]
        pl_marker.color.g = color[1]
        pl_marker.color.b = color[2]
        pl_marker.lifetime = rospy.Duration.from_sec(1200)
        pl_marker.action = Marker.ADD
        pl_marker.points = tuple(point_list)
        if not topic in self.marker_pub:
            self.marker_pub[topic] = rospy.Publisher(topic, Marker, queue_size=100)
        self.marker_pub[topic].publish(pl_marker)
