#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from ultralytics import YOLO
from rostopic_2_cv import *

# foco_px * distancia_real_entre_keypoints [px*m]. Calibrar: a uma altura Z conhecida,
# ALT_SCALE = Z * d_px medido. Sem isso, dist_2_alt nao esta em metros.
ALT_SCALE = 250

def dist_2_pts(x1, y1, x2, y2):
	return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find(frame, model):
	h_img, w_img = frame.shape[:2]
	results = model(np.ascontiguousarray(frame), verbose=False, imgsz=640, conf=0.8)

	if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
		return None

	boxes = results[0].boxes
	if len(boxes) == 0:
		return None

	best_idx = int(boxes.conf.argmax())

	kps = results[0].keypoints.xy[best_idx].cpu().numpy()  # (2, 2)
	if kps.shape[0] < 2:
		rospy.logwarn("No keypoints detected inside the bbox")
		return None

	center = np.array([kps[0][0], kps[0][1]], dtype=float)
	
	corners_quantity = kps.shape[0] - 1
	corners = [np.array([kps[i][0], kps[i][1]], dtype=float) for i in range(1, corners_quantity + 1)]

	x, y = int(center[0]), int(center[1])

	d = dist_2_pts(center[0], center[1], corners[0][0], corners[0][1])
	if d < 10:
		return None

	cv2.circle(frame, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)
	cv2.circle(frame, (int(corners[0][0]), int(corners[0][1])), 6, (0, 165, 255), -1)
	# Altura ~ inversamente proporcional a distancia (px) entre os keypoints:
	#   d_px = (foco_px * dist_real_kp) / Z  ->  Z = ALT_SCALE / d_px
	#   ALT_SCALE = foco_px * dist_real_kp [px*m], calibrar empiricamente.
	dist_2_alt = np.clip(ALT_SCALE / d, 0.0, 6.0)

	return x, y, dist_2_alt



def main():
	rospy.init_node('base_reader', anonymous=False)

	publish_image = rospy.get_param('~publish_image', True)
	rate_hz = rospy.get_param('~rate', 8)


	model_path = rospy.get_param('~model',
							  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_models/yolo11n/best_ncnn_model'))

	model = YOLO(model_path, task='pose')
	rospy.loginfo("Modelo YOLO Pose carregado: %s", model_path)

	def _on_shutdown():
		rospy.loginfo("Saida forcada para liberar threads do NCNN/torch")
		os._exit(0)
	rospy.on_shutdown(_on_shutdown)

	bridge = CvBridge() if publish_image else None
	pose_pub = rospy.Publisher('base_reader/pose', Point, queue_size=1)
	calib_pub = rospy.Publisher('base_reader/image_calib/compressed', CompressedImage, queue_size=1) if publish_image else None

	gaugecam = WebcamNode()
	rospy.loginfo("webcam_opencv rodando. Escutando /camera/image_raw...")

	rate = rospy.Rate(rate_hz)

	rospy.loginfo("No base_reader iniciado (model=%s, publish_image=%s, rate=%dHz)", model_path, publish_image, rate_hz)

	while not rospy.is_shutdown():
		frame = gaugecam.cv_image

		if frame is None:
			rospy.logwarn("Sem frame da camera. Tentando novamente...")
			rate.sleep()
			continue

		clean_frame = frame.copy()
		result = find(frame, model)

		if result is None:
			cv2.putText(frame, "Nao detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		else:
			x, y, dist_2_alt = result
			rospy.loginfo("Detectado: x=%d y=%d alt~%.3f", x, y, dist_2_alt)
			msg = Point()
			msg.x = float(x)
			msg.y = float(y)
			msg.z = float(dist_2_alt)
			pose_pub.publish(msg)
			
		if calib_pub is not None:
			_, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
			msg = CompressedImage()
			msg.header.stamp = rospy.Time.now()
			msg.format = 'jpeg'
			msg.data = jpeg.tobytes()
			calib_pub.publish(msg)

		rate.sleep()

if __name__=='__main__':
	main()
