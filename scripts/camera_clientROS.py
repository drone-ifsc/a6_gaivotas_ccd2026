#!/usr/bin/env python3
import cv2
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoResponse
import rospy
from cv_bridge import CvBridge, CvBridgeError
import yaml
import rospkg
import os
import numpy as np
import requests
import socket
import struct
from threading import Thread
import time

class FlaskCamera:
    def __init__(self, host_ip, port=8080, udp=False, udp_port=8081):
        self.host_ip = host_ip
        self.port = port
        self.udp = udp
        self.udp_port = udp_port
        self.stream_url = f"http://{host_ip}:{port}/stream"
        self.snapshot_url = f"http://{host_ip}:{port}/snapshot"
        self.frame = None
        self.stopped = False

    def start_stream(self):
        target = self.update_frame_udp if self.udp else self.update_frame
        self._thread = Thread(target=target)
        self._thread.daemon = True
        self._thread.start()
        return self
    def update_frame(self):
        while not self.stopped:
            try:
                stream = requests.get(self.stream_url, stream=True, timeout=10)
                bytes_data = bytes()

                # Flush stale buffer: read fast without decoding until frames
                # arrive with < 200ms gap (we're live)
                flush_start = time.time()
                last_frame_time = None
                flushing = True

                for chunk in stream.iter_content(chunk_size=65536):
                    if self.stopped:
                        return
                    bytes_data += chunk

                    # Drain to latest complete JPEG frame
                    jpg = None
                    while True:
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1 and b > a:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                        else:
                            break

                    if jpg is None:
                        continue

                    now = time.time()
                    if flushing:
                        if last_frame_time is not None and (now - last_frame_time) > 0.05:
                            # Frames arriving slowly = buffer caught up, we're live
                            flushing = False
                        last_frame_time = now
                        continue

                    self.frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
            except Exception as e:
                rospy.logwarn_throttle(5, "Stream error: %s", e)
    
    def update_frame_udp(self):
        MAX_FRAG = 60000
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        server = (self.host_ip, self.udp_port)

        # Subscribe
        sock.sendto(b"SUBSCRIBE", server)

        # Renew subscription every 5 seconds
        def keepalive():
            while not self.stopped:
                sock.sendto(b"SUBSCRIBE", server)
                time.sleep(5)
        Thread(target=keepalive, daemon=True).start()

        fragments = {}  # frame_id -> {frag_idx: data}
        while not self.stopped:
            try:
                pkt, _ = sock.recvfrom(65536)
            except socket.timeout:
                sock.sendto(b"SUBSCRIBE", server)
                continue
            if len(pkt) < 8:
                continue
            frame_id, frag_idx, total_frags = struct.unpack('>IHH', pkt[:8])
            data = pkt[8:]

            if frame_id not in fragments:
                fragments[frame_id] = {}
            fragments[frame_id][frag_idx] = data

            if len(fragments[frame_id]) == total_frags:
                jpg = b"".join(fragments[frame_id][i] for i in range(total_frags))
                fragments = {}  # drop all pending, only latest matters
                self.frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

        sock.sendto(b"UNSUBSCRIBE", server)
        sock.close()

    def get_frame(self):
        # Snapshot approach for single images
        response = requests.get(self.snapshot_url)
        return cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )
    
    def stop(self):
        self.stopped = True

class CameraHandler:
    def __init__(self):
        rospy.init_node('camera_handler')
        rospack = rospkg.RosPack()
        default_ost = os.path.join(rospack.get_path('gauge_reader'), 'camera_correction/imx219_3280x2464/ost.yaml')

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.frame_id = rospy.get_param('~frame_id', 'camera_frame')
        self.ost_file = rospy.get_param('~ost_file', default_ost)
        self.color_correction = rospy.get_param('~correction', False)
        
        # Initialize camera capture
        use_udp = rospy.get_param('~use_udp', True)
        udp_port = rospy.get_param('~udp_port', 8081)
        rospy.loginfo("Initializing camera (udp=%s)...", use_udp)
        self.camera = FlaskCamera("127.0.0.1", udp=use_udp, udp_port=udp_port).start_stream()
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        self.jpeg_quality = rospy.get_param('~jpeg_quality', 50)
        self.publish_compressed = rospy.get_param('~compressed', True)

        # Publishers for image and camera info
        self.image_pub = rospy.Publisher(self.image_topic, Image, queue_size=10)
        if self.publish_compressed:
            self.compressed_pub = rospy.Publisher(self.image_topic + '/compressed', CompressedImage, queue_size=1)
        self.camera_info_pub = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=10)
        
        # Create and publish camera info (dummy values for demonstration)
        self.camera_info = self.get_camera_info()

        # Service for camera calibration (cameracalibrator.py uses this to save results)
        camera_ns = self.camera_info_topic.rsplit('/camera_info', 1)[0]
        rospy.Service(camera_ns + '/set_camera_info', SetCameraInfo, self.set_camera_info_cb)

        rospy.loginfo("Camera Handler initialized with topics: %s and %s", self.image_topic, self.camera_info_topic)
        while not rospy.is_shutdown():
            self.publish_camera_data()
            time.sleep(0.033)  # Adjust the sleep time as needed
    def publish_camera_data(self):
        # Simulate capturing an image (replace with actual camera capture code)
        try:
            img = self.camera.frame
            if img is None:
                rospy.logwarn("No image captured from camera.")
                return

            # Color correction for OV5647 pink tint
            if self.color_correction:
                img = img.astype(np.float32)
                img[:, :, 2] *= 0.7   # Reduce Red:   89/152 ≈ 0.59
                img[:, :, 1] *= 0.8   # Reduce Green 
                img[:, :, 0] *= 0.70   # Boost Blue:   89/77  ≈ 1.16            img = np.clip(img, 0, 255).astype(np.uint8)
            
                img = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.convertScaleAbs(img, alpha=1.9, beta=-68)


            if hasattr(self, 'map1'):
                img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

            img = cv2.rotate(img, cv2.ROTATE_180) #cv2.ROTATE_90_CLOCKWISE
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = self.frame_id
            
            # Publish raw image
            self.image_pub.publish(ros_image)

            # Publish compressed image (JPEG) for WiFi transmission
            if self.publish_compressed:
                _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                compressed = CompressedImage()
                compressed.header = ros_image.header
                compressed.format = 'jpeg'
                compressed.data = jpeg.tobytes()
                self.compressed_pub.publish(compressed) 

            self.camera_info.header = ros_image.header
            #Publish the camera info
            self.camera_info_pub.publish(self.camera_info)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def get_camera_info(self):
        try:
            with open(self.ost_file, 'r') as file:
                data = yaml.safe_load(file)
                camera_info = CameraInfo()

                camera_info.width = data['image_width']
                camera_info.height = data['image_height']

                camera_info.distortion_model = data['distortion_model']

                camera_info.D = data["distortion_coefficients"]["data"]  # Distortion coefficients
                camera_info.K = data["camera_matrix"]["data"]  # Intrinsic parameters
                camera_info.R = data["rectification_matrix"]["data"] # Rotation matrix
                camera_info.P = data["projection_matrix"]["data"] # Projection matrix

                self.K_mat = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
                self.D_mat = np.array(data["distortion_coefficients"]["data"])
                h, w = data['image_height'], data['image_width']
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.K_mat, self.D_mat, None, self.K_mat, (w, h), cv2.CV_16SC2)
                
                camera_info.binning_x = 0
                camera_info.binning_y = 0

                camera_info.roi.x_offset = 0
                camera_info.roi.y_offset = 0
                camera_info.roi.height = 0
                camera_info.roi.width = 0
                camera_info.roi.do_rectify = False

                return camera_info
            print(data)
        except FileNotFoundError:
            rospy.logerr("Check ost.yaml! File not found: %s", self.ost_file)
            return CameraInfo()
        except yaml.YAMLError as exc:
            rospy.logerr("Check ost.yaml! YAML error: %s", exc)
            return CameraInfo()

    def set_camera_info_cb(self, req):
        self.camera_info = req.camera_info
        try:
            data = {
                'image_width': req.camera_info.width,
                'image_height': req.camera_info.height,
                'distortion_model': req.camera_info.distortion_model,
                'distortion_coefficients': {'data': list(req.camera_info.D), 'rows': 1, 'cols': len(req.camera_info.D)},
                'camera_matrix': {'data': list(req.camera_info.K), 'rows': 3, 'cols': 3},
                'rectification_matrix': {'data': list(req.camera_info.R), 'rows': 3, 'cols': 3},
                'projection_matrix': {'data': list(req.camera_info.P), 'rows': 3, 'cols': 4},
            }
            with open(self.ost_file, 'w') as f:
                yaml.dump(data, f)
            rospy.loginfo("Camera calibration saved to %s", self.ost_file)
            return SetCameraInfoResponse(success=True, status_message="Saved")
        except Exception as e:
            rospy.logerr("Failed to save calibration: %s", e)
            return SetCameraInfoResponse(success=False, status_message=str(e))

    def __del__(self):
        self.camera.stop()
        # Wait for stream thread to finish
        if hasattr(self.camera, '_thread') and self.camera._thread is not None:
            self.camera._thread.join(timeout=2.0)
        try:
            rospy.loginfo("Camera Handler node shutting down.")
        except Exception:
            pass

if __name__ == '__main__':
    try:
        camera_handler_node = CameraHandler()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        camera_handler_node.__del__()
        pass
