#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = None
        rospy.Subscriber("/camera/image_raw", Image, self.callback)

    def callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("Erro na conversão: %s", e)

if __name__ == "__main__":
    rospy.init_node("webcam_node", anonymous=True)
    WebcamNode()
    rospy.loginfo("Nó webcam_opencv rodando. Escutando /camera/image_raw...")
    rospy.spin()

