import os
import json
import time
import socket
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import zmq
import math
import torch


# =========================================================
# Shared latest frame (CameraHub)
# =========================================================
class FrameHub:
    """Single-owner camera reader; other threads only read latest frame."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0

    def set(self, frame):
        with self._lock:
            self._frame = frame
            self._ts = time.time()

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

# ========================================================
# set new object to record the results of YOLO detect
# ========================================================
class SharedYOLO:
    def __init__(self):
        self._lock = threading.Lock()
        self._dets = []  # list of dicts

    def set(self, dets):
        with self._lock:
            self._dets = dets

    def get(self):
        with self._lock:
            return list(self._dets)

class ZMQPublisher:
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 100)  # Set send high water mark
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ZMQ] Publishing on port {port}")
        self._frame_count = 0
    
    def send_frame(self, frame, topic="annotated_frame"):
        """Send video frame"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.socket.send_multipart([topic.encode(), buffer.tobytes()])
        self._frame_count += 1
    
    def send_json(self, data, topic="detections"):
        """Send JSON data"""
        self.socket.send_multipart([topic.encode(), json.dumps(data).encode()])
    
    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass
        print(f"[ZMQ] Publisher closed after {self._frame_count} frames sent")

# =========================================================
# Part 1: Lid state server (from detect_and_sent2.py)
# =========================================================
class SharedState:
    """thread-safe shared tuple state"""
    def __init__(self, init_state=(1, 0.0, 0.0, 5, 0.0, 0.0)):
        self._lock = threading.Lock()
        self._state = init_state

    def set(self, state_tuple):
        with self._lock:
            self._state = state_tuple

    def get(self):
        with self._lock:
            return self._state


class VisionServer:
    def __init__(self, shared_state: SharedState, host: str = "0.0.0.0", port: int = 9000, send_hz: float = 5.0):
        self.host = host
        self.port = port
        self.send_hz = max(0.5, float(send_hz))
        self.shared_state = shared_state

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)
        self.server_sock.settimeout(1.0)

        self.is_running = False
        self._conn_lock = threading.Lock()
        self._current_conn: Optional[socket.socket] = None

        print(f"[LID-Server] Listening on {self.host}:{self.port}  send_hz={self.send_hz}")

    def start(self) -> None:
        self.is_running = True
        print("[LID-Server] Start accept loop")

        try:
            while self.is_running:
                try:
                    conn, addr = self.server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                print(f"[LID-Server] Client connected: {addr}")
                with self._conn_lock:
                    self._current_conn = conn

                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()

        finally:
            self.close()

    def handle_client(self, conn: socket.socket, addr):
        period = 1.0 / self.send_hz
        try:
            while self.is_running:
                state = self.shared_state.get()
                msg = ",".join(str(x) for x in state) + "\n"
                conn.sendall(msg.encode("utf-8"))
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print(f"[LID-Server] Client disconnected: {addr}")

    def close(self) -> None:
        self.is_running = False
        with self._conn_lock:
            if self._current_conn is not None:
                try:
                    self._current_conn.close()
                except OSError:
                    pass
                self._current_conn = None
        try:
            self.server_sock.close()
        except OSError:
            pass
        print("[LID-Server] Closed")


# ---------------------------
# Lid detection utils
# ---------------------------
def crop(img, roi):
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

def lid_is_closed_by_template(roi_bgr, ref_bgr, corr_thr=0.4):
    roi_g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    roi_g = cv2.GaussianBlur(roi_g, (3,3), 0)
    ref_g = cv2.GaussianBlur(ref_g, (3,3), 0)
    res = cv2.matchTemplate(roi_g, ref_g, cv2.TM_CCOEFF_NORMED)
    score = float(res.max())
    return (score >= corr_thr), score

def state_from_lr(L_open, R_open):
    if L_open and R_open: return "ALL_OPEN"
    if (not L_open) and (not R_open): return "ALL_CLOSED"
    if L_open and (not R_open): return "LEFT_OPEN"
    return "RIGHT_OPEN"

def waffle_stuck_score(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    mask = (h >= 10) & (h <= 35) & (s >= 40) & (v >= 60)
    brown_ratio = float(mask.mean())
    mean_v = float(v.mean())
    return brown_ratio, mean_v

def decide_stuck(brown_ratio, mean_v, brown_thr=0.08, v_thr=105):
    return (brown_ratio > brown_thr) or (mean_v > v_thr)

def waffle_bottom_center(roi_bgr, v_dark_thr=85, min_ratio=0.08):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mask = (v > v_dark_thr).astype(np.uint8)
    non_black_ratio = float(mask.mean())
    if non_black_ratio < min_ratio:
        return False, (None, None), non_black_ratio
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < 1e-6:
        return False, (None, None), non_black_ratio
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    return True, (float(cx), float(cy)), non_black_ratio

def draw_box(img, roi, color, text):
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, text, (x, max(20, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, text, (x, max(20, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def make_side_code(is_open: bool, lid_stuck: bool, bottom_has_waffle: bool, side: str) -> int:
    base = 0 if side.upper() == 'L' else 4
    if not is_open: return base + 1
    if lid_stuck:   return base + 2
    if bottom_has_waffle: return base + 3
    return base + 4


# =========================================================
# Part 2: YOLO socket sender
# =========================================================
class RobotCommander:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.connect()

    def connect(self) -> bool:
        self.close()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"[YOLO-Socket] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[YOLO-Socket] Connection failed: {e}")
            self.sock = None
            return False

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def send_goal(self, signal):
        if not self.sock:
            print("[YOLO-Socket] Not connected. Reconnecting...")
            if not self.connect():
                return False
        try:
            # FIX: add newline delimiter to prevent TCP message merging
            msg = ", ".join(map(str, signal)) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            return True
        except Exception as e:
            print(f"[YOLO-Socket] Send failed: {e}")
            self.sock = None
            return False


def transform_xyxy(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:     #座標轉換，將影像座標轉成Isaac-Sim的座標系統
    # transform from image coordinate to isaac-sim coordinate
    # new_x1 = -45 + (y1 - 58) * 0.1264044943820225
    # new_y1 = -46.72057 + (x1 - 312) * 0.125

    # new_width = (y2 - y1) * 0.1264044943820225   #新的寬度
    # new_height = (x2 - x1) * 0.125     #新的高度

    # 將杯子定位點從左上角移動到杯底中心
    # center_x = new_x1 + new_width * (1-0.5)
    # (1-x_offset)為偏移比例，為杯底中心距離左上角的距離占整個杯寬的比例
    # center_y = new_y1 + new_height * (1-0.18)   # (1-y_offset)為偏移比例，為杯底中心距離左上角的距離占整個杯寬的比例

    new_x1 = -45 + (y1 - 26) * 0.1243093922651934
    new_y1 = -46.72057 + (x1 - 260) * 0.1238816242257398

    new_width_offset = ((y2 - y1) - 29) * 0.1243093922651934
    new_height_offset = ((x2 - x1) - 25.5) * 0.1238816242257398

    # 將杯子定位點從左上角移動到杯底中心
    center_x = new_x1 + new_width_offset
    center_y = new_y1 + new_height_offset
    
    return center_x, center_y, x2, y2

# =========================================================
# Threads
# =========================================================
def camera_loop(framehub: FrameHub, stop_evt: threading.Event, cam_conf: dict):
    idx = int(cam_conf.get("index", 0))
    backend_name = str(cam_conf.get("backend", "CAP_DSHOW"))
    backend = cv2.CAP_DSHOW if backend_name == "CAP_DSHOW" else 0

    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    w = int(cam_conf.get("width", 1920))
    h = int(cam_conf.get("height", 1080))
    fps = int(cam_conf.get("fps", 30))
    fourcc_str = str(cam_conf.get("fourcc", "MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))

    print(f"[Camera] Opened index={idx} backend={backend_name} {w}x{h}@{fps} FOURCC={fourcc_str}")

    try:
        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            framehub.set(frame)
    finally:
        cap.release()
        print("[Camera] Released")


def lid_loop(framehub: FrameHub, shared_lid: SharedState, shared_yolo: SharedYOLO, stop_evt: threading.Event, conf: dict, debug_ui=True):
    cfg = conf["roi"]["lid"]
    waffle_cfg = conf["roi"]["waffle"]
    thr = conf["threshold"]

    corr_thr = float(thr["lid_close"]["corr_thr"])
    brown_thr = float(thr["stuck_waffle"]["brown_thr"])
    v_thr = float(thr["stuck_waffle"]["v_thr"])
    v_dark_thr = float(thr["bottom_waffle"]["v_dark_thr"])
    bottom_min_ratio = float(thr["bottom_waffle"]["min_ratio"])

    refL = cv2.imread(conf["ref"]["close_ref_left"])
    refR = cv2.imread(conf["ref"]["close_ref_right"])
    if refL is None or refR is None:
        raise RuntimeError("close_ref images not found")

    win = "Integrated (Lid+YOLO)"
    if debug_ui:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_ts = 0.0
    while not stop_evt.is_set():
        frame, ts = framehub.get()
        if frame is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts

        roiL = crop(frame, cfg["left"])
        roiR = crop(frame, cfg["right"])

        L_closed, L_score = lid_is_closed_by_template(roiL, refL, corr_thr=corr_thr)
        R_closed, R_score = lid_is_closed_by_template(roiR, refR, corr_thr=corr_thr)
        L_open, R_open = (not L_closed), (not R_closed)
        lid_state = state_from_lr(L_open, R_open)

        stuck_L = stuck_R = False
        if L_open:
            ul = crop(frame, waffle_cfg["upper_left"])
            br, mv = waffle_stuck_score(ul)
            stuck_L = decide_stuck(br, mv, brown_thr=brown_thr, v_thr=v_thr)
        if R_open:
            ur = crop(frame, waffle_cfg["upper_right"])
            br, mv = waffle_stuck_score(ur)
            stuck_R = decide_stuck(br, mv, brown_thr=brown_thr, v_thr=v_thr)

        bottom_has_L = bottom_has_R = False
        cL = cR = (None, None)

        if L_open and (not stuck_L):
            bottom_has_L, cL, _ = waffle_bottom_center(roiL, v_dark_thr=v_dark_thr, min_ratio=bottom_min_ratio)
        if R_open and (not stuck_R):
            bottom_has_R, cR, _ = waffle_bottom_center(roiR, v_dark_thr=v_dark_thr, min_ratio=bottom_min_ratio)

        L_code = make_side_code(L_open, stuck_L, bottom_has_L, 'L')
        R_code = make_side_code(R_open, stuck_R, bottom_has_R, 'R')

        L_dx = L_dy = 0.0
        R_dx = R_dy = 0.0
        if L_code == 3 and cL[0] is not None:
            _, _, wL, hL = cfg["left"]
            L_dx = float(cL[1] - hL / 2.0 - 1.0)
            L_dy = float(cL[0] - wL / 2.0 + 1.0)
        if R_code == 7 and cR[0] is not None:
            _, _, wR, hR = cfg["right"]
            R_dx = float(cR[1] - hR / 2.0 - 1.0)
            R_dy = float(cR[0] - wR / 2.0 - 1.0)

        msg_tuple = (int(L_code), round(L_dx, 1), round(L_dy, 1),
                     int(R_code), round(R_dx, 1), round(R_dy, 1))
        shared_lid.set(msg_tuple)

        # cv2.imwrite("debug_frame.jpg", frame)
        if debug_ui:
            vis = frame.copy()

            yolo_dets = shared_yolo.get()
            for d in yolo_dets:
                x1,y1,x2,y2 = map(int, d["xyxy"])
                cls = d["cls"]
                conf_val = d["conf"]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(vis, f"YOLO {cls} {conf_val:.2f}", (x1, max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            cv2.putText(vis, f"LID_STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 6)
            cv2.putText(vis, f"LID_STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(vis, f"LID_CODE: {msg_tuple}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
            cv2.putText(vis, f"LID_CODE: {msg_tuple}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            draw_box(vis, cfg["left"],  (0,255,0) if L_open else (0,0,255), f"L {'OPEN' if L_open else 'CLOSE'} score={L_score:.2f}")
            draw_box(vis, cfg["right"], (0,255,0) if R_open else (0,0,255), f"R {'OPEN' if R_open else 'CLOSE'} score={R_score:.2f}")

            # cv2.imwrite("debug_frame.jpg", vis)
            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                stop_evt.set()
                break


def yolo_loop(framehub: FrameHub, shared_yolo: SharedYOLO, stop_evt: threading.Event, conf: dict):
    yconf = conf["yolo"]

    model_path = yconf["model"]
    conf_th = float(yconf.get("conf", 0.25))
    imgsz = int(yconf.get("imgsz", 640))
    send_hz = float(yconf.get("send_hz", 5.0))
    sleep_t = 1.0 / max(send_hz, 0.5)
    enable_send = bool(yconf.get("enable_send", True))

    # FIX: Read all host/port pairs from config instead of hardcoding
    # Config format:  "targets": [{"host":"192.168.1.33","port":9999}, {"host":"192.168.1.34","port":9999}]
    # Falls back to old single host/port if "targets" not present
    targets = yconf.get("targets", None)
    if targets is None:
        # Backward compatible: build target list from old single-host keys
        host1 = str(yconf.get("host", "192.168.1.34"))
        port1 = int(yconf.get("port", 9999))
        host2 = str(yconf.get("host2", "192.168.1.33"))
        port2 = int(yconf.get("port2", 9999))
        targets = [
            {"host": host1, "port": port1},
            #{"host": host2, "port": port2},
        ]

    # FIX: Accepted YOLO classes are configurable; defaults to [0, 1, 2]
    accepted_classes = set(yconf.get("accepted_classes", [0, 1, 2]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    print(f"[YOLO] Using device: {device}")

    # Print model class names so operator can verify
    print(f"[YOLO] Model classes: {model.names}")

    commanders = []
    if enable_send:
        for t in targets:
            h = str(t["host"])
            p = int(t["port"])
            cmd = RobotCommander(h, p)
            commanders.append(cmd)
            print(f"[YOLO] Target registered: {h}:{p}")
    
    print(f"[YOLO] model={model_path} conf={conf_th} imgsz={imgsz} "
          f"accepted_classes={accepted_classes} targets={len(commanders)} @ {send_hz}Hz")

    MOVE_THRES_PX = 10.0
    last_sent_px: Dict[int, Optional[Tuple[float, float]]] = {}

    last_ts = 0.0
    while not stop_evt.is_set():
        frame, ts = framehub.get()
        if frame is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts

        results = model.predict(source=frame, conf=conf_th, imgsz=imgsz, verbose=False)
        r = results[0]

        best: Dict[int, Tuple[float, Tuple[float, float, float, float]]] = {}
        dets_for_ui = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                c = float(c); k = int(k)
                if c < conf_th:
                    continue

                # Save ALL detections for UI display (no class filter)
                dets_for_ui.append({
                    "cls": k,
                    "conf": c,
                    "xyxy": (float(x1), float(y1), float(x2), float(y2))
                })

                # FIX: Use configurable accepted_classes instead of hardcoded (0, 1)
                if k not in accepted_classes:
                    print(f"[YOLO] Ignored class {k} (not in accepted_classes={accepted_classes})")
                    continue

                prev = best.get(k)
                if prev is None or c > prev[0]:
                    best[k] = (c, (float(x1), float(y1), float(x2), float(y2)))

        # FIX: Always update shared_yolo, even when no detections (clears stale boxes)
        shared_yolo.set(dets_for_ui)

        # Send best detection per accepted class
        for class_id in accepted_classes:
            if class_id not in best:
                continue

            _, (x1, y1, x2, y2) = best[class_id]

            cur_px = (float(x1), float(y1))
            prev_px = last_sent_px.get(class_id)

            moved = True
            if prev_px is not None:
                dx = cur_px[0] - prev_px[0]
                dy = cur_px[1] - prev_px[1]
                dist = math.hypot(dx, dy)
                moved = dist >= MOVE_THRES_PX
            
            if not moved:
                continue

            tx1, ty1, _, _ = transform_xyxy(x1, y1, x2, y2)

            signal = [
                class_id,
                float(tx1),
                float(ty1),
                90.0,
                -180.0,
                0.0,
                0.0,
            ]

            if enable_send:
                for cmd in commanders:
                    # print(f"[YOLO] Sending class {class_id} to {cmd.host}:{cmd.port} -> signal: {signal}")
                    cmd.send_goal(signal)
                time.sleep(0.05)

            last_sent_px[class_id] = cur_px

        time.sleep(sleep_t)

    for cmd in commanders:
        cmd.close()
    print("[YOLO] Stopped")

def zmq_stream_loop(framehub: FrameHub, shared_yolo: SharedYOLO, shared_lid: SharedState,
                    stop_evt: threading.Event, zmq_pub: ZMQPublisher, stream_hz: float = 10.0, conf: dict = None):
    """Stream fully annotated frames and detection data via ZMQ"""
    period = 1.0 / stream_hz
    last_ts = 0.0

    lid_cfg = conf["roi"]["lid"] if (conf and "roi" in conf and "lid" in conf["roi"]) else None

    while not stop_evt.is_set():
        frame, ts = framehub.get()
        if frame is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts

        yolo_dets = shared_yolo.get()
        lid_tuple = shared_lid.get()
        L_code, L_dx, L_dy, R_code, R_dx, R_dy = lid_tuple

        L_open = L_code in (2, 3, 4)
        R_open = R_code in (6, 7, 8)
        lid_state_str = state_from_lr(L_open, R_open)

        vis = frame.copy()
        for d in yolo_dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cls = d["cls"]
            confv = d["conf"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, f"YOLO {cls} {confv:.2f}", (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        try:
            cv2.putText(vis, f"LID_STATE: {lid_state_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(vis, f"LID_STATE: {lid_state_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"LID_CODE: {(L_code, round(L_dx,1), round(L_dy,1), R_code, round(R_dx,1), round(R_dy,1))}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
            cv2.putText(vis, f"LID_CODE: {(L_code, round(L_dx,1), round(L_dy,1), R_code, round(R_dx,1), round(R_dy,1))}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception:
            pass

        if lid_cfg:
            try:
                lx, ly, lw, lh = lid_cfg["left"]
                rx, ry, rw, rh = lid_cfg["right"]
                cv2.rectangle(vis, (lx, ly), (lx+lw, ly+lh), (0, 255, 0) if L_open else (0, 0, 255), 2)
                cv2.putText(vis, f"L {'OPEN' if L_open else 'CLOSE'}", (lx, max(20, ly-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0, 255, 0) if R_open else (0, 0, 255), 2)
                cv2.putText(vis, f"R {'OPEN' if R_open else 'CLOSE'}", (rx, max(20, ry-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception:
                pass

        try:
            zmq_pub.send_frame(vis, topic="annotated_frame")
            data = {
                "timestamp": ts,
                "yolo_detections": yolo_dets,
                "lid_state_tuple": lid_tuple,
                "lid_state": lid_state_str
            }
            zmq_pub.send_json(data, topic="detections")
        except Exception:
            pass

        time.sleep(period)

    print("[ZMQ] Stream stopped")

# =========================================================
# Main
# =========================================================
def main():
    CONFIG_PATH = "config_yolo.json"
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError("config_yolo.json not found")

    conf = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))

    for k in ("roi", "threshold", "ref", "camera", "server", "yolo", "zmq"):
        if k not in conf:
            raise RuntimeError(f"config_yolo.json missing key: {k}")

    stop_evt = threading.Event()
    framehub = FrameHub()
    shared_yolo = SharedYOLO()

    shared_lid = SharedState(init_state=(1, 0.0, 0.0, 5, 0.0, 0.0))
    srv = conf["server"]
    lid_server = VisionServer(
        shared_state=shared_lid,
        host=str(srv.get("host", "0.0.0.0")),
        port=int(srv.get("port", 9000)),
        send_hz=float(srv.get("send_hz", 5.0)),
    )

    zmq_conf = conf.get("zmq", {})
    zmq_port = int(zmq_conf.get("port", 9999))
    zmq_hz = float(zmq_conf.get("stream_hz", 10.0))
    zmq_pub = ZMQPublisher(port=zmq_port)

    t_server = threading.Thread(target=lid_server.start, daemon=True)
    t_cam = threading.Thread(target=camera_loop, args=(framehub, stop_evt, conf["camera"]), daemon=True)
    t_yolo = threading.Thread(target=yolo_loop, args=(framehub, shared_yolo, stop_evt, conf), daemon=True)
    t_lid  = threading.Thread(target=lid_loop,  args=(framehub, shared_lid, shared_yolo, stop_evt, conf), kwargs={"debug_ui": True}, daemon=True)
    t_zmq = threading.Thread(target=zmq_stream_loop, 
                         args=(framehub, shared_yolo, shared_lid, stop_evt, zmq_pub, zmq_hz, conf), 
                         daemon=True)

    t_server.start()
    t_cam.start()
    t_lid.start()
    t_yolo.start()
    t_zmq.start()

    print("[MAIN] Running. Press 'q' on the window to quit.")

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    finally:
        stop_evt.set()

        try:
            lid_server.close()
        except Exception:
            pass

        try:
            zmq_pub.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        for t in (t_yolo, t_lid, t_cam, t_server, t_zmq):
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

        print("[MAIN] Exit.")

if __name__ == "__main__":
    main()