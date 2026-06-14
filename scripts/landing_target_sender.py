#!/usr/bin/env python3
"""ROS Noetic node: base_reader pose -> MAVLink LANDING_TARGET via MAVROS.

Subscribes to the target center published by base_reader as a
``geometry_msgs/Point`` on ``base_reader/pose`` where:
    x, y -> target center in image pixels
    z    -> altitude/distance estimate [m] (base_reader dist_2_alt)

Camera intrinsics are taken automatically from ``sensor_msgs/CameraInfo``
(``/camera/camera_info``). The pixel offset is converted into angular offsets
and republished as a ``mavros_msgs/LandingTarget`` on
``/mavros/landing_target/raw``. The MAVROS ``landing_target`` plugin forwards it
to the FCU as a MAVLink LANDING_TARGET message (e.g. ArduPilot precision landing).

Output rate control:
    New detections are forwarded immediately, so if base_reader runs faster than
    ``~rate`` the output follows that higher rate. If detections arrive slower
    than ``~rate``, the last coordinate is repeated to keep the output at a
    floor of ``~rate`` Hz (default 10 Hz). The achieved output rate is logged.

Private parameters:
    ~input_topic       (str): source Point topic (default 'base_reader/pose')
    ~camera_info_topic (str): CameraInfo topic (default '/camera/camera_info')
    ~target_num        (int): LANDING_TARGET target id (default 0)
    ~rate              (float): minimum output rate in Hz (default 10.0)
"""
import rospy
import threading
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
from mavros_msgs.msg import LandingTarget
import math


def marker_size_rad(larg, alt, distance):
    marker_w = larg         # largura real [m]
    marker_h = alt          # altura real [m]
    D        = max(distance, 1e-3)   # mesma distância enviada no campo distance [m]

    size_x = 2 * math.atan(marker_w / (2 * D))   # rad
    size_y = 2 * math.atan(marker_h / (2 * D))   # rad
    return (size_x, size_y)


class TargetPosePublisher:
    # Dimensões reais do alvo [m] usadas para calcular o tamanho angular.
    MARKER_W = 0.18
    MARKER_H = 0.18

    def __init__(self):
        rospy.init_node('target_pose_publisher', anonymous=True)

        # Parameters
        self.target_num = rospy.get_param('~target_num', 0)          # ID do alvo
        input_topic = rospy.get_param('~input_topic', 'base_reader/pose')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.rate_hz = float(rospy.get_param('~rate', 10.0))         # piso de taxa de saída [Hz]
        self.min_period = 1.0 / self.rate_hz

        # Intrínsecos da câmera: preenchidos a partir do CameraInfo (matriz K).
        self.fx = self.fy = self.cx = self.cy = None

        # Estado compartilhado entre callbacks e timers
        self.lock = threading.Lock()
        self.last_point = None
        self.last_pub_time = rospy.Time(0)
        self.pub_count = 0       # contadores p/ o monitor de taxa
        self.repeat_count = 0

        # Publisher para o plugin landing_target do MAVROS (envia LANDING_TARGET para a FCU)
        self.lt_pub = rospy.Publisher('/mavros/landing_target/raw', LandingTarget, queue_size=10)

        # Intrínsecos da câmera (companion topic de /camera/image_raw)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        # Subscriber: centro do alvo (em pixels) + estimativa de altura, vindo do base_reader
        rospy.Subscriber(input_topic, Point, self.pose_callback)

        # Timer de piso: repete a última coordenada se nada novo chegou dentro do período
        self.floor_timer = rospy.Timer(rospy.Duration(self.min_period), self.floor_cb)
        # Monitor: loga a taxa de saída a cada 2s
        self.monitor_timer = rospy.Timer(rospy.Duration(2.0), self.monitor_cb)

        rospy.loginfo("Target Pose Publisher initialized. %s -> /mavros/landing_target/raw "
                      "@ >= %.1f Hz (intrinsics from %s)",
                      input_topic, self.rate_hz, camera_info_topic)

    def camera_info_callback(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1] (row-major 3x3)
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def pose_callback(self, msg):
        # Nova detecção: guarda e publica imediatamente (deixa passar taxas > ~rate).
        with self.lock:
            self.last_point = msg
        self._publish(msg)

    def floor_cb(self, event):
        # Se nada foi publicado dentro do período, repete a última coordenada (piso de taxa).
        with self.lock:
            point = self.last_point
            elapsed = (rospy.Time.now() - self.last_pub_time).to_sec()
        if point is not None and elapsed >= self.min_period:
            self._publish(point, repeated=True)

    def monitor_cb(self, event):
        with self.lock:
            n, r = self.pub_count, self.repeat_count
            self.pub_count = self.repeat_count = 0
        rospy.loginfo("LANDING_TARGET out: %.1f Hz (%d repetidos)", n / 2.0, r)

    def _publish(self, point, repeated=False):
        if self.fx is None:
            rospy.logwarn_throttle(2.0, "Sem CameraInfo ainda; aguardando intrínsecos da câmera")
            return

        px, py, distance = point.x, point.y, point.z

        # Pixel offset em relação ao ponto principal -> deslocamento angular [rad].
        # Pode ser necessário inverter sinais conforme a orientação de montagem da câmera.
        angle_x = math.atan2(px - self.cx, self.fx)
        angle_y = math.atan2(py - self.cy, self.fy)

        sizes = marker_size_rad(self.MARKER_W, self.MARKER_H, distance)

        lt = LandingTarget()
        lt.header.stamp = rospy.Time.now()
        lt.target_num = self.target_num
        lt.frame = LandingTarget.BODY_NED          # enum próprio do mavros (=9), corpo do drone
        lt.angle = [angle_x, angle_y]              # [rad] deslocamento angular X, Y
        lt.distance = distance                     # [m]
        lt.size = [sizes[0], sizes[1]]             # [rad] (firmwares novos tendem a ignorar)
        lt.type = LandingTarget.VISION_FIDUCIAL

        self.lt_pub.publish(lt)
        with self.lock:
            self.last_pub_time = rospy.Time.now()
            self.pub_count += 1
            if repeated:
                self.repeat_count += 1


if __name__ == '__main__':
    try:
        TargetPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
