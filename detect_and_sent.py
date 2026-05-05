import os
import json
import time
import socket
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------
# [A] server part
# ---------------------------

class SharedState:
    """執行緒安全的共享狀態：存最新 msg_tuple"""
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
        self.send_hz = max(0.5, float(send_hz))  # 最慢 0.5Hz，避免 0 除
        self.shared_state = shared_state

        # TCP server socket
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)
        self.server_sock.settimeout(1.0)

        self.is_running = False
        self._current_conn: Optional[socket.socket] = None
        self._current_addr: Optional[Tuple[str, int]] = None
        self._conn_lock = threading.Lock()

        print(f"[INIT] Server listening on {self.host}:{self.port}  send_hz={self.send_hz}")

    def start(self) -> None:
        """accept loop"""
        self.is_running = True
        print("[START] TCPServer is running.")

        try:
            while self.is_running:
                try:
                    conn, addr = self.server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                print(f"[CONNECT] Client connected from {addr}")
                with self._conn_lock:
                    self._current_conn = conn
                    self._current_addr = addr

                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()

        except KeyboardInterrupt:
            print("\n[STOP] KeyboardInterrupt, shutting down.")
        finally:
            self.close()

    def handle_client(self, conn: socket.socket, addr):
        period = 1.0 / self.send_hz
        try:
            while self.is_running:
                state = self.current_state()
                # 送 "1,0.0,0.0,5,0.0,0.0\n"
                msg = ",".join(str(x) for x in state) + "\n"
                conn.sendall(msg.encode("utf-8"))
                # for debug
                # print(f"[SEND] to {addr}: {state}")
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print(f"[DISCONNECT] {addr}")

    def current_state(self):
        return self.shared_state.get()

    def close(self) -> None:
        if self.is_running:
            self.is_running = False

        with self._conn_lock:
            if self._current_conn is not None:
                try:
                    self._current_conn.close()
                except OSError:
                    pass
                self._current_conn = None
                self._current_addr = None

        try:
            self.server_sock.close()
        except OSError:
            pass

        print("[CLOSE] Server socket closed.")


# ---------------------------
# [B] detect part
# ---------------------------

#ROI_PATH = "roi_config.json"
#THR_PATH = "thresholds.json"
#WAFFLE_ROI_PATH = "waffle_roi_config.json"
CONFIG_PATH = "config.json"

USE_CAMERA = True
CAM_INDEX = 0

def crop(img, roi):
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

def lid_is_closed_by_template(roi_bgr, ref_bgr, corr_thr=0.4):
    # template matching
    roi_g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    roi_g = cv2.GaussianBlur(roi_g, (3,3), 0)
    ref_g = cv2.GaussianBlur(ref_g, (3,3), 0)

    res = cv2.matchTemplate(roi_g, ref_g, cv2.TM_CCOEFF_NORMED)
    score = float(res.max())
    is_closed = (score >= corr_thr)
    return is_closed, score

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
    # 非黑：v > v_dark_thr
    mask = (v > v_dark_thr).astype(np.uint8)

    non_black_ratio = float(mask.mean())
    #print(f"non_black_ratio: {non_black_ratio}")
    if non_black_ratio < min_ratio:
        return False, (None, None), non_black_ratio, (mask * 255)

    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < 1e-6:
        return False, (None, None), non_black_ratio, (mask * 255)

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return True, (float(cx), float(cy)), non_black_ratio, (mask * 255)

def draw_box(img, roi, color, text):
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, text, (x, max(20, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, text, (x, max(20, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def make_side_code(is_open: bool, lid_stuck: bool, bottom_has_waffle: bool, side: str) -> int:
    base = 0 if side.upper() == 'L' else 4  # L +0, R +4
    if not is_open:
        return base + 1
    if lid_stuck:
        return base + 2
    if bottom_has_waffle:
        return base + 3
    return base + 4


# ---------------------------
# [C] main：開 webcam + 偵測 + 更新 shared_state（server 端就會送出去）
# ---------------------------

if __name__ == "__main__":

    # 1) 讀 ROI/threshold 設定
    if not os.path.exists(CONFIG_PATH): raise RuntimeError("config.json not found")
    conf = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))

    cfg = conf["roi"]["lid"]            # cfg["left"], cfg["right"]
    waffle_cfg = conf["roi"]["waffle"]  # waffle_cfg["upper_left"], ["upper_right"]
    thr = conf["threshold"]             # thr["lid_close"]["corr_thr"] ...


    # 2) close refs
    raise_if_missing = False
    if raise_if_missing:
        pass

    refL = cv2.imread(conf["ref"]["close_ref_left"])
    refR = cv2.imread(conf["ref"]["close_ref_right"])

    if refL is None or refR is None:
        raise RuntimeError("close_ref_left.png / close_ref_right.png not found ")

    # 3) 啟動共享狀態 + TCP Server（daemon thread）
    shared = SharedState(init_state=(1, 0.0, 0.0, 5, 0.0, 0.0))
    srv = conf["server"]
    server = VisionServer(
        shared_state=shared,
        host=str(srv.get("host", "0.0.0.0")),
        port=int(srv.get("port", 9000)),
        send_hz=float(srv.get("send_hz", 5.0)),
    )
    #server = VisionServer(shared_state=shared, host="0.0.0.0", port=9000, send_hz=5.0)
    t_server = threading.Thread(target=server.start, daemon=True)
    t_server.start()

    # 4) 開相機（沿用你已確認可用的 CAP_DSHOW + 1920x1080）
    cap = None
    cam = conf["camera"]
    USE_CAMERA = bool(cam.get("use_camera", True))
    CAM_INDEX = int(cam.get("index", 0))

    backend_name = str(cam.get("backend", "CAP_DSHOW"))
    backend = cv2.CAP_DSHOW if backend_name == "CAP_DSHOW" else 0
    
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAM_INDEX, backend)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(cam.get("width", 1920)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 1080)))
        cap.set(cv2.CAP_PROP_FPS,          int(cam.get("fps", 30)))

        fourcc_str = str(cam.get("fourcc", "MJPG"))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    #parameters setting
    corr_thr = float(thr["lid_close"]["corr_thr"])
    brown_thr = float(thr["stuck_waffle"]["brown_thr"])
    v_thr = float(thr["stuck_waffle"]["v_thr"])
    v_dark_thr = float(thr["bottom_waffle"]["v_dark_thr"])
    bottom_min_ratio = float(thr["bottom_waffle"]["min_ratio"])

    print("[INFO] Keys: q=quit, s=save screenshot")

    try:
        while True:
            ret, img = cap.read()
            if not ret or img is None:
                print("camera read failed")
                break

            # --- 1) open/close ---
            roiL = crop(img, cfg["left"])
            roiR = crop(img, cfg["right"])
            L_closed, L_diff = lid_is_closed_by_template(roiL, refL, corr_thr=corr_thr)
            R_closed, R_diff = lid_is_closed_by_template(roiR, refR, corr_thr=corr_thr)
            
            L_open = not L_closed
            R_open = not R_closed
            lid_state = state_from_lr(L_open, R_open)

            # --- 2) 上蓋黏餅（open 才算） ---
            stuck_L = stuck_R = False
            if L_open:
                roi_ul = crop(img, waffle_cfg["upper_left"])
                L_brown, L_v = waffle_stuck_score(roi_ul)
                stuck_L = decide_stuck(L_brown, L_v, brown_thr=brown_thr, v_thr=v_thr)
            if R_open:
                roi_ur = crop(img, waffle_cfg["upper_right"])
                R_brown, R_v = waffle_stuck_score(roi_ur)
                stuck_R = decide_stuck(R_brown, R_v, brown_thr=brown_thr, v_thr=v_thr)

            # --- 3) 下烤盤有無鬆餅 + 中心 ---
            bottom_has_L = bottom_has_R = False
            cL = cR = (None, None)

            if L_open and (not stuck_L):
                #bottom_has_L, cL, ratioL, maskL = waffle_bottom_center(roiL, v_dark_thr=V_DARK_THR, min_ratio=BOTTOM_MIN_RATIO)
                bottom_has_L, cL, ratioL, maskL = waffle_bottom_center(roiL, v_dark_thr=v_dark_thr, min_ratio=bottom_min_ratio)
            if R_open and (not stuck_R):
                #bottom_has_R, cR, ratioR, maskR = waffle_bottom_center(roiR, v_dark_thr=V_DARK_THR, min_ratio=BOTTOM_MIN_RATIO)
                bottom_has_R, cR, ratioR, maskR = waffle_bottom_center(roiR, v_dark_thr=v_dark_thr, min_ratio=bottom_min_ratio)

            # --- 4) 產生 code + dx/dy（你原本的規則） ---
            L_code = make_side_code(L_open, stuck_L, bottom_has_L, 'L')
            R_code = make_side_code(R_open, stuck_R, bottom_has_R, 'R')

            # dx/dy：只在 code=3(L) 或 code=7(R) 有意義
            L_dx = L_dy = 0.0
            R_dx = R_dy = 0.0

            if L_code == 3 and cL[0] is not None:
                _, _, wL, hL = cfg["left"]
                L_dx = float(cL[0] - wL/2.0)
                L_dy = float(cL[1] - hL/2.0)

            if R_code == 7 and cR[0] is not None:
                _, _, wR, hR = cfg["right"]
                R_dx = float(cR[0] - wR/2.0)
                R_dy = float(cR[1] - hR/2.0)

            # 小數 1 位
            L_dx_o = round(L_dx, 1)
            L_dy_o = round(L_dy, 1)
            R_dx_o = round(R_dx, 1)
            R_dy_o = round(R_dy, 1)

            msg_tuple = (L_code, L_dx_o, L_dy_o, R_code, R_dx_o, R_dy_o)

            # 每幀更新 shared state，server 端會自動送出去
            shared.set(msg_tuple)

            # --- 視覺化 ---
            vis = img.copy()
            cv2.putText(vis, f"STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 6)
            cv2.putText(vis, f"STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            cv2.putText(vis, f"CODE: {msg_tuple}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
            cv2.putText(vis, f"CODE: {msg_tuple}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            draw_box(vis, cfg["left"],  (0,255,0) if L_open else (0,0,255), f"L {'OPEN' if L_open else 'CLOSE'} score={L_diff:.2f}")
            draw_box(vis, cfg["right"], (0,255,0) if R_open else (0,0,255), f"R {'OPEN' if R_open else 'CLOSE'} score={R_diff:.2f}")

            cv2.imshow("Camera Infer + TCP Server", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("debug_camera.jpg", vis)
                print("[OK] Saved debug_camera.jpg")

    finally:
        try:
            server.close()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] bye")
