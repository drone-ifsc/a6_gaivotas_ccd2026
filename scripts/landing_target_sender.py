#!/usr/bin/env python3
"""ROS Noetic node: base_reader pose -> MAVLink LANDING_TARGET via MAVROS.

Subscribes to the target center published by base_reader as a
``geometry_msgs/Point`` on ``base_reader/pose`` where:
    x, y -> target center in image pixels
    z    -> altitude estimate [m] (base_reader dist_2_alt, needs ALT_SCALE calib)

Camera intrinsics come from ``sensor_msgs/CameraInfo`` (``/camera/camera_info``).

Position-based output (workaround for the MAVROS plugin):
    The MAVROS landing_target plugin HARD-CODES position_valid=1 in landtarget_cb,
    so ArduPilot takes the position branch instead of the angle branch. We therefore
    back-project the pixel center into a 3D line-of-sight vector in body-FRD
    (x=forward, y=right, z=down) and send it as pose.position, with frame =
    MAV_FRAME_BODY_FRD (12). angle_x/angle_y are still filled as a fallback (used if
    the link drops the v2 extension fields and position_valid degrades to 0).

    NB: the plugin applies ENU->NED to pose.position before sending, so we
    pre-compensate (pose = (right, forward, -down)) to get the intended FRD vector.

Private parameters:
    ~input_topic       (str): source Point topic (default 'base_reader/pose')
    ~camera_info_topic (str): CameraInfo topic (default '/camera/camera_info')
    ~target_num        (int): LANDING_TARGET target id (default 0)
    ~rate              (float): minimum output rate in Hz (default 10.0)
    ~forward_sign      (float): +1/-1, flip if drone moves away along forward axis
    ~right_sign        (float): +1/-1, flip if drone moves away along right axis
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
    # mavros encaminha req->frame cru -> usamos o valor MAVLink direto.
    MAV_FRAME_BODY_FRD = 12

    # Dimensões reais do alvo [m] usadas para calcular o tamanho angular.
    MARKER_W = 1
    MARKER_H = 1

    def __init__(self):
        rospy.init_node('target_pose_publisher', anonymous=True)

        # Parameters
        self.target_num = rospy.get_param('~target_num', 0)          # ID do alvo
        input_topic = rospy.get_param('~input_topic', 'base_reader/pose')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.rate_hz = float(rospy.get_param('~rate', 10.0))         # piso de taxa de saída [Hz]
        self.min_period = 1.0 / self.rate_hz
        self.forward_sign = float(rospy.get_param('~forward_sign', 1.0))
        self.right_sign = float(rospy.get_param('~right_sign', 1.0))

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
                      "@ >= %.1f Hz (intrinsics from %s, BODY_FRD position mode)",
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

        px, py = point.x, point.y
        h = max(point.z, 0.2)   # altura/profundidade estimada [m] (evita vetor nulo)

        # Back-projeta o pixel -> raio na câmera (OpenCV: x dir., y baixo, z frente/óptico).
        u = (px - self.cx) / self.fx
        v = (py - self.cy) / self.fy

        # Câmera apontada para baixo -> body FRD. Profundidade ao longo do eixo óptico = h.
        #   imagem-cima  -> frente (-v)
        #   imagem-direita -> direita (u)
        #   eixo óptico  -> baixo (z)
        fwd   = self.forward_sign * (-v) * h
        right = self.right_sign   * (u)  * h
        down  = h

        slant = math.sqrt(fwd * fwd + right * right + down * down)

        # Ângulos (fallback se o link cair p/ v1 e as extensões/position_valid sumirem).
        angle_x = math.atan2(px - self.cx, self.fx)
        angle_y = math.atan2(py - self.cy, self.fy)

        sizes = marker_size_rad(self.MARKER_W, self.MARKER_H, slant)

        lt = LandingTarget()
        lt.header.stamp = rospy.Time.now()
        lt.target_num = self.target_num
        lt.frame = self.MAV_FRAME_BODY_FRD         # 12, encaminhado cru pelo mavros
        lt.angle = [angle_x, angle_y]              # [rad] fallback angular
        lt.distance = slant                        # [m] alcance até o alvo
        lt.size = [sizes[0], sizes[1]]             # [rad] (firmwares novos ignoram)
        lt.type = LandingTarget.VISION_FIDUCIAL

        # mavros força position_valid=1 e aplica ENU->NED em pose.position.
        # Pré-compensa para que lt.(x,y,z) saiam como (fwd, right, down) em BODY_FRD:
        #   lt.x = pose.y ; lt.y = pose.x ; lt.z = -pose.z
        lt.pose.position.x = right
        lt.pose.position.y = fwd
        lt.pose.position.z = -down
        lt.pose.orientation.w = 1.0                # evita quaternion nulo (NaN no transform)

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
