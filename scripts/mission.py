#!/usr/bin/env python3
import math
import threading
from collections import deque
import rospy
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, ParamSet, CommandHome
from mavros_msgs.msg import PositionTarget, State, ParamValue, StatusText
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_geometry_msgs
import tf2_ros
from webcam_opencv.msg import ObjectDetected
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import os

alt = 2

TARGET_COORDS = [
    #(2, 0),
    #(2, 3)
    (4, 0) 
]

class MissionFrame:
    """Mantem um frame fixo no ponto de inicio da missao."""

    def __init__(self):
        self.local_frame = rospy.get_param("~local_frame", "map")
        self.frame_id = rospy.get_param("~mission_frame", "mission_start")
        self.ready = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

    def capture(self, local_position, yaw_enu):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.local_frame
        transform.child_frame_id = self.frame_id
        transform.transform.translation.x = local_position.x
        transform.transform.translation.y = local_position.y
        transform.transform.translation.z = local_position.z

        q = quaternion_from_euler(0.0, 0.0, yaw_enu)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]

        self.static_tf_broadcaster.sendTransform(transform)
        if hasattr(self.tf_buffer, "set_transform_static"):
            self.tf_buffer.set_transform_static(transform, "mission_node")

        self.ready = True
        rospy.loginfo(
            f"mission_start capturado em {self.local_frame}: "
            f"x={local_position.x:.2f}, y={local_position.y:.2f}, "
            f"z={local_position.z:.2f}, yaw={math.degrees(yaw_enu):.1f}deg"
        )

    def to_local_pose(self, x_forward, y_right, z_up):
        mission_pose = PoseStamped()
        mission_pose.header.stamp = rospy.Time(0)
        mission_pose.header.frame_id = self.frame_id
        mission_pose.pose.position.x = x_forward
        mission_pose.pose.position.y = -y_right  # API usa direita positiva; TF usa esquerda positiva.
        mission_pose.pose.position.z = z_up
        mission_pose.pose.orientation.w = 1.0

        transform = self.tf_buffer.lookup_transform(
            self.local_frame,
            self.frame_id,
            rospy.Time(0),
            rospy.Duration(0.2)
        )
        return tf2_geometry_msgs.do_transform_pose(mission_pose, transform)

    def to_mission_position(self, local_position):
        local_pose = PoseStamped()
        local_pose.header.stamp = rospy.Time(0)
        local_pose.header.frame_id = self.local_frame
        local_pose.pose.position = local_position
        local_pose.pose.orientation.w = 1.0

        transform = self.tf_buffer.lookup_transform(
            self.frame_id,
            self.local_frame,
            rospy.Time(0),
            rospy.Duration(0.1)
        )
        mission_pose = tf2_geometry_msgs.do_transform_pose(local_pose, transform)
        return Point(
            mission_pose.pose.position.x,
            -mission_pose.pose.position.y,
            mission_pose.pose.position.z
        )

class DroneMission:
    def __init__(self):
        rospy.init_node("mission_node", anonymous=True)

        # Servicos MAVROS
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/takeoff")
        rospy.wait_for_service('/mavros/param/set')
        rospy.wait_for_service('/mavros/cmd/set_home')

        self.arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.takeoff_srv = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)
        self.set_home_srv = rospy.ServiceProxy('/mavros/cmd/set_home', CommandHome)

        # Servico opcional: salva o frame atual em mission_captures/
        # Nao bloqueia inicializacao se o servico ainda nao subiu.
        self.save_img_srv = rospy.ServiceProxy('save_img', Trigger)

        # Subscribers para topicos MAVROS
        self.current_pose   = None
        self.current_local_pose = None
        self.state          = State()
        self.mission_frame = MissionFrame()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.pose_callback(rospy.wait_for_message("/mavros/local_position/pose", PoseStamped))
        self.state_callback(rospy.wait_for_message("/mavros/state", State))

        # Publisher para posicao local
        self.pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.setpoint_raw_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)

        # Evento: PrecLand reportou "target found" via STATUSTEXT da FCU
        self._precland_event = threading.Event()
        rospy.Subscriber('/mavros/statustext/recv', StatusText, self._on_statustext)

        # Handler de shutdown para pouso seguro
        rospy.on_shutdown(self.safe_shutdown)

    def pose_callback(self, msg):
        """Recebe posicao do MAVROS"""
        self.current_local_pose = msg.pose.position
        self.current_orientation = msg.pose.orientation

        # Conversao de quaternion para Roll Pitch Yaw
        q = self.current_orientation
        quaternion = [q.x, q.y, q.z, q.w]
        self.current_roll, self.current_pitch, self.current_yaw_enu = euler_from_quaternion(quaternion)
        self.current_yaw = self.current_yaw_enu - math.pi/2

        if self.mission_frame.ready:
            try:
                self.current_pose = self.mission_frame.to_mission_position(msg.pose.position)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.current_pose = Point(
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                )
        else:
            self.current_pose = Point(
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            )

    def state_callback(self, msg):
        self.state = msg

    def extended_state_callback(self, msg):
        self.extended_state = msg


    def arm_and_takeoff(self, altitude=2.0):
        self.set_mode_srv(custom_mode="GUIDED")
        rospy.loginfo("Mode: GUIDED")

        rospy.loginfo("Armando...")
        self.arming_srv(True)

        rospy.loginfo(f"Decolando para {altitude}m...")
        self.takeoff_srv(altitude=altitude)
        timeout = rospy.Time.now().to_sec() + 10
        while self.current_pose.z <= 0.80 * (altitude):
            if timeout < rospy.Time.now().to_sec():
                rospy.logwarn(f"Timeout de chegar na altura")
                break
            rospy.loginfo(f"Altitude: {self.current_pose.z:.2f}m")
            rospy.sleep(1)

    def body_to_enu(self, x_body, y_body):
        """Converte coordenadas do frame do corpo para ENU"""
        yaw_rad = self.current_yaw
        east  = -x_body * math.sin(yaw_rad) + y_body * math.cos(yaw_rad)
        north =  x_body * math.cos(yaw_rad) + y_body * math.sin(yaw_rad)
        return east, north

    def enu_to_body(self, east, north):
        """Converte coordenadas ENU para frame do corpo"""
        yaw_rad = self.current_yaw
        x_body = -east * math.sin(yaw_rad) + north * math.cos(yaw_rad)
        y_body =  east * math.cos(yaw_rad) + north * math.sin(yaw_rad)
        return x_body, y_body

    def capture_mission_start_frame(self):
        """Fixa o frame mission_start na posicao/yaw atuais antes da decolagem."""
        if self.current_local_pose is None or self.current_orientation is None:
            rospy.logwarn("Sem pose local para capturar mission_start")
            return False

        self.mission_frame.capture(self.current_local_pose, self.current_yaw_enu)
        self.current_pose = self.mission_frame.to_mission_position(self.current_local_pose)
        return True

    def send_position(self, x, y, z, tol=0.4, wait=20):
        """Publica setpoint absoluto no frame mission_start. x=frente, y=direita, z=cima.

        Publica continuamente a 10Hz ate que o drone chegue dentro de `tol` metros
        ou que `wait` segundos passem. ArduCopter GUIDED precisa de stream continuo.
        """
        timeout = rospy.Time.now().to_sec() + wait

        # Caso especial: mission_start ainda nao capturado — usa frame local MAVROS, sem loop
        if not self.mission_frame.ready:
            rospy.logwarn_throttle(2.0, "mission_start ainda nao capturado; usando frame local MAVROS")
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation = self.current_orientation
            self.pos_pub.publish(msg)
            return

        # Loop ate chegar ou estourar timeout, publicando setpoint a cada tick
        rate = rospy.Rate(10)
        try:
            local_msg = self.mission_frame.to_local_pose(x, y, z)
            local_msg.header.stamp = rospy.Time.now()
            local_msg.header.frame_id = self.mission_frame.local_frame
            self.pos_pub.publish(local_msg)

            while not rospy.is_shutdown():
                dist = math.sqrt(
                    (x - self.current_pose.x) ** 2 +
                    (y - self.current_pose.y) ** 2 +
                    (z - self.current_pose.z) ** 2
                )

                if dist < tol:
                    rospy.loginfo(f"Chegou na posicao ({x:.2f}, {y:.2f}, {z:.2f})")
                    return

                if rospy.Time.now().to_sec() > timeout:
                    rospy.logwarn(f"Timeout chegando em ({x:.2f}, {y:.2f}, {z:.2f}); dist atual={dist:.2f}m")
                    return

                rate.sleep()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn(f"Falha ao transformar setpoint mission_start->{self.mission_frame.local_frame}: {exc}")
            return


    def send_body_offset(self, dx, dy, dz=0.0):
        """Publica setpoint relativo no body frame (FRAME_BODY_OFFSET_NED = 9).

        dx = frente (+) / tras (-)
        dy = esquerda (+) / direita (-)
        dz = cima (+) / baixo (-)

        Apesar do frame MAVLink ser BODY_OFFSET_NED, este topico MAVROS recebe
        o vetor em convencao ROS/base_link e converte para NED internamente.
        Diferente de send_position: nao depende de current_pose, FCU aplica
        """
        target = PositionTarget()
        target.header.stamp = rospy.Time.now()
        target.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        target.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY |
                            PositionTarget.IGNORE_VZ | PositionTarget.IGNORE_AFX |
                            PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                            PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE)
        target.position.x = dx
        target.position.y = dy
        target.position.z = dz
        self.setpoint_raw_pub.publish(target)

    def reset_home(self):
        resp = self.set_home_srv(current_gps=True, latitude=0, longitude=0, altitude=0)
        rospy.loginfo(f"Home definida na posicao atual: {resp.success}")
        
    def _on_statustext(self, msg):
        # ArduPilot emite STATUSTEXT "PrecLand: Target Found" quando o alvo e adquirido.
        text = msg.text.lower()

        # Info: PrecLand: Target Found
        # Info: PrecLand: Init Complete
        
        if 'PrecLand' in text and 'Complete' in text:
            self._precland_event.set()

    def wait_precland_ready(self, timeout=5.0):
        """Espera (poll com timeout) o proximo "target found" do PrecLand.

        Chame logo apos disparar o PrecLand (ex.: ao entrar em LAND). Limpa o
        evento internamente, entao aguarda um found NOVO dentro de `timeout`.
        ArduPilot emite o STATUSTEXT na transicao de aquisicao, nao de forma
        continua. Retorna True se ficou pronto, False no timeout.
        """
        self._precland_event.clear()
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown():
            if self._precland_event.is_set():
                rospy.loginfo("PrecLand pronto (target found)")
                return True
            if rospy.Time.now() > deadline:
                rospy.logwarn("PrecLand nao ficou pronto em %.1fs", timeout)
                return False
            rate.sleep()
        return False

    def _save_capture(self):
        """Chama save_img_service para salvar o frame atual em mission_captures/.
        Falha silenciosa se o servico nao estiver disponivel."""
        try:
            resp = self.save_img_srv()
            if resp.success:
                rospy.loginfo("Captura salva: %s", resp.message)
            else:
                rospy.logwarn("save_img falhou: %s", resp.message)
        except rospy.ServiceException as e:
            rospy.logwarn("save_img servico indisponivel: %s", e)

    def execute_mission(self):
        global alt
        READ_TIMEOUT = 7.0
        DESCEND_STEP = 0.3
        MAX_ATTEMPTS = 6

        try:
            self.reset_home()
            rospy.sleep(3)
            self.capture_mission_start_frame()

            self.arm_and_takeoff(alt)
            rospy.loginfo("Pairando por 3 segundos")
            rospy.sleep(3)

            for i in TARGET_COORDS:
                    
                self.send_position(i[0], i[1], alt)
                if self.wait_precland_ready(5.0):
                    rospy.loginfo("PrecLand activated normally")
                else:
                    self.send_body_offset(0, 0, -0.5)
                    if self.wait_precland_ready(5.0):
                        rospy.loginfo("PrecLand activated from second try")
                    else:
                        rospy.loginfo("No PrecLand")
                self.set_mode_srv(custom_mode="LAND")
                rospy.sleep(20)

                self.arm_and_takeoff(alt)
                rospy.loginfo("Pairando por 3 segundos")
                rospy.sleep(3)

            rospy.loginfo("Retornando para a origem")
            self.send_position(0, 0, alt)

        except rospy.ROSInterruptException:
            rospy.loginfo("Missao interrompida pelo usuario")

    def safe_shutdown(self):
        rospy.loginfo("Shutdown iniciado. Tentando pousar...")
        try:
            rospy.loginfo("Retornando para a origem")
            self.send_position(0, 0, alt)

            self.set_mode_srv(custom_mode="LAND")
            self.arming_srv(False)
            rospy.loginfo("Pouso e desarme enviados com sucesso")

        except rospy.ServiceException:
            rospy.logwarn("Servicos MAVROS indisponiveis durante shutdown")
           
        rospy.loginfo("Saida forcada para encerrar o no")
        os._exit(0)

if __name__ == "__main__":
    mission = DroneMission()
    mission.execute_mission()
