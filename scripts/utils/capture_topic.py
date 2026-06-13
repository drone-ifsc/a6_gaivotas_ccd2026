#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import os
import time


def get_image_topics():
    try:
        output = subprocess.check_output(['rostopic', 'list'], text=True)
        topics = output.strip().split('\n')
        image_topics = []
        for t in topics:
            if '/compressed' in t:
                image_topics.append((t, 'compressed'))
            elif any(k in t for k in ['image', 'camera', 'webcam', 'frame']):
                image_topics.append((t, 'raw'))
        return image_topics
    except Exception as e:
        print(f"rostopic list failed: {e}")
        return []


def pick(label, options):
    print(f"\n{label}")
    for i, (name, proto) in enumerate(options):
        print(f"  {i+1}. {name}  [{proto}]")
    print("  0. Enter manually")
    while True:
        try:
            choice = int(input("Choice: "))
            break
        except ValueError:
            pass
    return choice


def main():
    print("=== ROS Image Capture ===")

    topics = get_image_topics()

    if topics:
        choice = pick("Available topics:", topics)
    else:
        print("No topics found.")
        choice = 0

    if choice == 0:
        topic = input("Topic: ").strip()
        print("Protocol:")
        print("  1. raw  (sensor_msgs/Image)")
        print("  2. compressed  (sensor_msgs/CompressedImage)")
        proto_choice = input("Choice [1/2]: ").strip()
        proto = 'compressed' if proto_choice == '2' else 'raw'
    else:
        topic, proto = topics[choice - 1]

    rospack = rospkg.RosPack()
    captures_dir = os.path.join(rospack.get_path('gauge_reader'), 'captures')
    os.makedirs(captures_dir, exist_ok=True)

    print(f"\nConnecting to '{topic}' [{proto}]...")
    print(f"Saving to: {captures_dir}")
    print("SPACE — capture frame | Q — quit\n")

    rospy.init_node('capture_topic', anonymous=True)
    bridge = CvBridge()
    frame_holder = {'img': None, 'stamp': None}

    def cb_raw(msg):
        try:
            frame_holder['img'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_holder['stamp'] = msg.header.stamp
        except Exception as e:
            rospy.logerr_throttle(5, "CvBridge error: %s", e)

    def cb_compressed(msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame_holder['img'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame_holder['stamp'] = msg.header.stamp
        except Exception as e:
            rospy.logerr_throttle(5, "Decode error: %s", e)

    if proto == 'compressed':
        rospy.Subscriber(topic, CompressedImage, cb_compressed, queue_size=1)
    else:
        rospy.Subscriber(topic, Image, cb_raw, queue_size=1)

    window = f"{topic} [{proto}]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    count = 0

    while not rospy.is_shutdown():
        img = frame_holder['img']
        stamp = frame_holder['stamp']

        if img is not None:
            display = img.copy()
            if stamp is not None:
                delay = time.time() - stamp.to_sec()
                cv2.putText(display, f"delay: {delay*1000:.0f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"saved: {count} | SPACE=capture Q=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow(window, display)
        else:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for frames...",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            cv2.imshow(window, blank)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and frame_holder['img'] is not None:
            filename = os.path.join(captures_dir, f"capture_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, frame_holder['img'])
            count += 1
            print(f"Saved [{count}]: {os.path.basename(filename)}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()