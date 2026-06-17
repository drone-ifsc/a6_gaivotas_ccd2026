#!/usr/bin/env python3
"""Servidor de camera hibrido.

Roda NO HOST (Raspberry Pi OS / Raspbian, RPi5, Python 3.10) e publica
quadros da camera CSI direto em topico ROS. Substitui o par
camera_server.py + camera_clientROS.py que usava TCP/UDP custom.

O roscore roda no container privileged com network_mode=host (ou rede
roteada para rede local). Como nao existe pacote apt de ROS para
Raspbian, o rospy aqui vem do projeto rospypi/simple (pure Python,
funciona em qualquer Python 3.x incluindo 3.10/3.11).

Instalacao no host:
    pip install --extra-index-url https://rospypi.github.io/simple/ \
        rospy rospkg sensor_msgs std_msgs geometry_msgs

    rpicam-apps ja vem com o sistema (libcamera + rpicam-vid).

Execucao:
    export ROS_MASTER_URI=http://<container_ip>:11311
    export ROS_IP=<host_ip>
    python3.10 camera_host_server.py

Parametros (via rosparam):
    ~publish_raw    bool — publicar Image decodificada      (default False)
    ~publish_jpeg   bool — publicar CompressedImage         (default True)
    ~width          int  — largura                          (default 3280)
    ~height         int  — altura                           (default 2464)
    ~framerate      int  — fps                              (default 4)
    ~quality        int  — qualidade JPEG 1-100             (default 60)
    ~undistort      bool — corrigir distorcao de lente      (default False)
    ~calib_file     str  — caminho para ost.yaml            (default: ost.yaml ao lado do script)
    ~publish_camera_info bool — publicar CameraInfo em /camera/camera_info (default True)
    ~frame_id       str  — frame_id de Image/CameraInfo     (default "camera")
    ~hsv_correct    bool — aplicar correcao HSV             (default False)
    ~hsv_h_shift    float — deslocamento de matiz [-180..180] (default 0)
    ~hsv_s_gain     float — ganho de saturacao [0..3]       (default 1.0)
    ~hsv_v_gain     float — ganho de valor/brilho [0..3]    (default 1.0)
"""
import os
import signal
import subprocess
import sys

import cv2
import numpy as np
import rospy
import yaml
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


def load_calibration(path):
    """Le ost.yaml e devolve (K, D, w_calib, h_calib)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).flatten()
    w = int(data["image_width"])
    h = int(data["image_height"])
    return K, D, w, h


def scale_K(K, w_calib, h_calib, w_actual, h_actual):
    """Escala a matriz K se a resolucao atual difere da resolucao de calibracao."""
    if (w_actual, h_actual) == (w_calib, h_calib):
        return K
    sx = w_actual / w_calib
    sy = h_actual / h_calib
    K = K.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def build_undistort_maps(K, D, w_actual, h_actual):
    """Cria maps para cv2.remap (K ja escalado para a resolucao atual)."""
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (w_actual, h_actual), cv2.CV_16SC2
    )
    return map1, map2


def build_camera_info(K, D, width, height, frame_id, rectified):
    """Monta um CameraInfo constante (so o header.stamp muda a cada frame).

    Se a imagem publicada ja foi retificada (undistort=True), D=0 e P usa a
    propria K, pois a distorcao foi removida usando K como nova camera matrix.
    Caso contrario, publica D real da calibracao (imagem ainda distorcida).
    """
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.D = [0.0] * 5 if rectified else [float(v) for v in D]
    info.K = [float(v) for v in K.flatten()]
    info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.P = [K[0, 0], K[0, 1], K[0, 2], 0.0,
              K[1, 0], K[1, 1], K[1, 2], 0.0,
              K[2, 0], K[2, 1], K[2, 2], 0.0]
    return info


def apply_hsv_correction(img_bgr, h_shift, s_gain, v_gain):
    """Aplica correcao HSV in-place onde possivel. h_shift em [-180..180], gains > 0."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if h_shift != 0:
        h = ((h.astype(np.int16) + int(h_shift / 2)) % 180).astype(np.uint8)
    if s_gain != 1.0:
        s = np.clip(s.astype(np.float32) * s_gain, 0, 255).astype(np.uint8)
    if v_gain != 1.0:
        v = np.clip(v.astype(np.float32) * v_gain, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def build_camera_cmd(width, height, framerate, quality):
    return [
        "rpicam-vid",
        "-t", "0",
        "--mode", f"{width}:{height}",
        "--width",  str(width),
        "--height", str(height),
        "--framerate", str(framerate),
        "--buffer-count", "1",
        "--codec", "mjpeg",
        "--quality", str(quality),
        "--shutter", "3000",     # фиксированная выдержка в микросекундах (8ms)
        "--gain", "1.0",         # фиксированный analog gain
        "--awb", "daylight",
        "--inline",
        "-o", "-",
    ]


def main():
    rospy.init_node("camera_host_server", anonymous=False)

    publish_raw  = rospy.get_param("~publish_raw",  False)
    publish_jpeg = rospy.get_param("~publish_jpeg", True)
    width        = rospy.get_param("~width",  1640)
    height       = rospy.get_param("~height", 1232)
    framerate    = rospy.get_param("~framerate", 5)
    quality      = rospy.get_param("~quality", 30)
    undistort    = rospy.get_param("~undistort",  True)
    default_calib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_correction/imx219_3280x2464/ost.yaml")
    calib_file   = rospy.get_param("~calib_file", default_calib)
    publish_camera_info = rospy.get_param("~publish_camera_info", True)
    frame_id     = rospy.get_param("~frame_id", "camera")

    hsv_correct  = rospy.get_param("~hsv_correct",  True)
    hsv_h_shift  = float(rospy.get_param("~hsv_h_shift", 0.0))
    hsv_s_gain   = float(rospy.get_param("~hsv_s_gain",  0.8))
    hsv_v_gain   = float(rospy.get_param("~hsv_v_gain",  0.8))
    if hsv_correct:
        rospy.loginfo("HSV correction ativado: h_shift=%.1f s_gain=%.2f v_gain=%.2f",
                      hsv_h_shift, hsv_s_gain, hsv_v_gain)

    if not (publish_raw or publish_jpeg):
        rospy.logfatal("publish_raw=False и publish_jpeg=False — нечего публиковать")
        sys.exit(1)

    # Calibracao: carrega uma vez. Necessaria para undistort e/ou CameraInfo.
    map1 = map2 = None
    camera_info_msg = None
    if undistort or publish_camera_info:
        try:
            K, D, w_calib, h_calib = load_calibration(calib_file)
            K = scale_K(K, w_calib, h_calib, width, height)
        except Exception as e:
            if undistort:
                rospy.logfatal("Falha ao carregar calibracao %s: %s", calib_file, e)
                sys.exit(1)
            rospy.logwarn("Sem calibracao (%s): CameraInfo desativado", e)
            publish_camera_info = False

    if undistort:
        map1, map2 = build_undistort_maps(K, D, width, height)
        rospy.loginfo("Undistort ativado. Calib: %s (calib %dx%d -> atual %dx%d)",
                      calib_file, w_calib, h_calib, width, height)

    if publish_camera_info:
        camera_info_msg = build_camera_info(K, D, width, height, frame_id, rectified=undistort)
        rospy.loginfo("CameraInfo ativado em /camera/camera_info (frame_id=%s, rectified=%s)",
                      frame_id, undistort)

    pub_jpeg = (rospy.Publisher("/camera/image_raw/compressed",
                                CompressedImage, queue_size=1)
                if publish_jpeg else None)
    pub_raw  = (rospy.Publisher("/camera/image_raw",
                                Image, queue_size=1)
                if publish_raw else None)
    pub_info = (rospy.Publisher("/camera/camera_info",
                                CameraInfo, queue_size=1)
                if publish_camera_info else None)

    cmd = build_camera_cmd(width, height, framerate, quality)
    rospy.loginfo("Starting: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    def cleanup(*_):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=2)
            except Exception:
                pass
        rospy.signal_shutdown("camera stopped")
        os._exit(0)

    rospy.on_shutdown(cleanup)
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    rospy.loginfo("Camera up. publish_jpeg=%s publish_raw=%s",
                  publish_jpeg, publish_raw)

    buf = bytearray()
    READ = 65536
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    while not rospy.is_shutdown():
        chunk = proc.stdout.read(READ)
        if not chunk:
            rospy.logwarn("rpicam-vid stdout closed")
            break
        buf.extend(chunk)

        # Извлекаем все полные JPEG из буфера (последний кадр публикуем)
        while True:
            soi = buf.find(SOI)
            eoi = buf.find(EOI, soi + 2) if soi != -1 else -1
            if soi == -1 or eoi == -1:
                break
            jpg = bytes(buf[soi:eoi + 2])
            del buf[:eoi + 2]

            stamp = rospy.Time.now()

            # Decodifica somente quando precisa (raw publish OU undistort OU hsv_correct)
            needs_decode = pub_raw is not None or undistort or hsv_correct
            img = None
            if needs_decode:
                arr = np.frombuffer(jpg, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if undistort:
                    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
                if hsv_correct:
                    img = apply_hsv_correction(img, hsv_h_shift, hsv_s_gain, hsv_v_gain)

            # Re-encode JPEG apenas se houve modificacao na imagem (undistort ou hsv)
            modified = undistort or hsv_correct

            if pub_info is not None:
                camera_info_msg.header.stamp = stamp
                pub_info.publish(camera_info_msg)

            if pub_jpeg is not None:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = frame_id
                msg.format = "jpeg"
                if modified:
                    ok, jpg_enc = cv2.imencode(".jpg", img,
                                               [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        continue
                    msg.data = jpg_enc.tobytes()
                else:
                    msg.data = jpg
                pub_jpeg.publish(msg)

            if pub_raw is not None:
                raw = Image()
                raw.header.stamp = stamp
                raw.header.frame_id = frame_id
                raw.height = img.shape[0]
                raw.width  = img.shape[1]
                raw.encoding = "bgr8"
                raw.is_bigendian = 0
                raw.step = img.shape[1] * 3
                raw.data = img.tobytes()
                pub_raw.publish(raw)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
