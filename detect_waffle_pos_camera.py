import cv2
import json
import os
import glob
import numpy as np
from collections import deque

ROI_PATH = "roi_config.json"              # detect lid open/close ROI
THR_PATH = "thresholds.json"              # detect black_thr
WAFFLE_ROI_PATH = "waffle_roi_config.json"

USE_CAMERA = True
CAM_INDEX = 0

refs = np.load("lid_close_refs.npz")
refL, refR = refs["refL"], refs["refR"]
thrL, thrR = float(refs["thrL"]), float(refs["thrR"])

def _normalize_gray(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    g = cv2.GaussianBlur(g, (5,5), 0)
    # 去掉亮度差：z-score normalize
    m, s = float(g.mean()), float(g.std()) + 1e-6
    g = (g - m) / s
    return g

def lid_is_closed_by_template(roi_bgr, ref_gray, corr_thr=0.55):
    """
    用模板相關係數判斷：越像金屬圓蓋(close_ref) => 越接近 1
    open 狀態(黑盤/鬆餅液/鬆餅/背景) => 相關會明顯下降
    回傳 (is_closed, score)
    """
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    ref = cv2.GaussianBlur(ref_gray, (5,5), 0)

    # ROI 跟 ref 尺寸若不同，這裡把 ROI resize 成 ref 尺寸（或反過來也可）
    if g.shape != ref.shape:
        g = cv2.resize(g, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    # 單點模板比對（同尺寸時 matchTemplate 只會吐 1 個值）
    res = cv2.matchTemplate(g, ref, cv2.TM_CCOEFF_NORMED)
    score = float(res[0,0])

    return (score >= corr_thr), score

def list_images(folder):
    exts=("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    paths=[]
    for e in exts:
        paths += glob.glob(os.path.join(folder,e))
    paths.sort()
    return paths

def crop(img, roi):
    x,y,w,h = roi
    return img[y:y+h, x:x+w].copy()

# --- 你原本的 lid feature：只用 black_ratio ---
def compute_black_ratio(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:,:,2], (5,5), 0)
    v_dark_thr = 85
    black_ratio = float((v < v_dark_thr).mean())
    return black_ratio

def waffle_bottom_center(roi_bgr, v_dark_thr=85, min_ratio=0.06, circle_margin=0.92):
    """
    在「下方烤盤 ROI」內，用亮度(V)把黑色烤盤排除，剩下的(鬆餅液/鬆餅)算中心點
    回傳:
      has_waffle: bool
      (cx, cy): ROI座標系中心點(float)
      non_black_ratio: 非黑比例
      mask: 用於debug顯示的mask(uint8 0/255)
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # 1) 非黑：亮度高於門檻
    mask = (v > v_dark_thr).astype(np.uint8)

    # 2) 圓形遮罩：只看烤盤圓內，避免邊框/外面背景干擾
    h, w = mask.shape
    cx0, cy0 = w / 2.0, h / 2.0
    r = int(min(w, h) * 0.5 * circle_margin)
    circ = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(circ, (int(cx0), int(cy0)), r, 1, -1)
    mask = (mask & circ).astype(np.uint8)

    non_black_ratio = float(mask.mean())
    if non_black_ratio < min_ratio:
        # 幾乎全黑 => 沒有鬆餅/鬆餅液
        return False, (None, None), non_black_ratio, (mask * 255)

    # 3) moments 算中心
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < 1e-6:
        return False, (None, None), non_black_ratio, (mask * 255)

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return True, (float(cx), float(cy)), non_black_ratio, (mask * 255)

def state_from_lr(L_open, R_open):
    if L_open and R_open: return "ALL_OPEN"
    if (not L_open) and (not R_open): return "ALL_CLOSED"
    if L_open and (not R_open): return "LEFT_OPEN"
    return "RIGHT_OPEN"

# --- 新增：上蓋黏餅偵測（ROI 內「黃褐色像素比例」）---
def waffle_stuck_score(roi_bgr):
    """
    回傳 (brown_ratio, mean_v)
    brown_ratio: HSV 中黃褐色像素比例（鬆餅顏色）
    mean_v: ROI 平均亮度，作為輔助
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # 黃褐色大致落在 H ~ [10, 35]（OpenCV Hue 0~179）
    # 再用 S,V 避免把灰白反光算進來
    mask = (h >= 10) & (h <= 35) & (s >= 40) & (v >= 60)
    brown_ratio = float(mask.mean())
    mean_v = float(v.mean())
    return brown_ratio, mean_v

def decide_stuck(brown_ratio, mean_v, brown_thr=0.08, v_thr=105):
    return (brown_ratio > brown_thr) or (mean_v > v_thr)

def draw_box(img, roi, color, text):
    x,y,w,h = roi
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(img, text, (x, max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, text, (x, max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def make_side_code(is_open: bool,
                   lid_stuck: bool,
                   bottom_has_waffle: bool,
                   side: str) -> int:
    """
    side: 'L' or 'R'
    Left  : close=1, lid_stuck=2, bottom_has=3, empty=4
    Right : close=5, lid_stuck=6, bottom_has=7, empty=8
    """
    base = 0 if side.upper() == 'L' else 4  # 左邊 +0, 右邊 +4

    if not is_open:
        return base + 1  # close

    # open
    if lid_stuck:
        return base + 2  # 上蓋黏著鬆餅

    # open & no lid stuck
    if bottom_has_waffle:
        return base + 3  # 下方有鬆餅
    else:
        return base + 4  # 下方沒有鬆餅（ready）

if __name__ == "__main__":
    INFER_FOLDER = r"./openlid"  # 你的測試資料夾（逐張按空白）

    if not os.path.exists(ROI_PATH): raise RuntimeError("roi_config.json not found")
    if not os.path.exists(THR_PATH): raise RuntimeError("thresholds.json not found")
    if not os.path.exists(WAFFLE_ROI_PATH): raise RuntimeError("waffle_roi_config.json not found (run setup_waffle_roi.py)")

    cfg = json.load(open(ROI_PATH,"r",encoding="utf-8"))
    thr = json.load(open(THR_PATH,"r",encoding="utf-8"))
    waffle_cfg = json.load(open(WAFFLE_ROI_PATH,"r",encoding="utf-8"))

    black_thr = float(thr.get("black_thr", 0.28))  # lid open 判斷門檻（你已驗證可用）

    # 黏餅門檻（先給一組保守值；之後你覺得太敏感/太鈍再調）
    BROWN_THR = 0.3
    V_THR = 105
    
    print("start to connect webcam")

    cap = None
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # 先用 1920x1080 @ 30 之類保守值
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 可加：指定 MJPG（常讓 BRIO 比較好談）
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    #paths = list_images(INFER_FOLDER)
    #if not paths: raise RuntimeError("No images in infer folder")

    #idx = 0
    #print("[INFO] Keys: space=next, b=back, s=save, q=quit")
    print("[INFO] Keys: q=quit, s=save screenshot")

    while True:
        if USE_CAMERA:
            ret, img = cap.read()
            if not ret or img is None:
                print("camera read failed")
                break

        # 1) resize to match ROI coordinate system
        #img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        #idx = max(0, min(idx, len(paths)-1))
        #p = paths[idx]
        #img = cv2.imread(p)
        #if img is None:
        #    idx += 1
        #    if idx >= len(paths): break
        #    continue

        # --- 1) 判斷 lids open/close（左右各自） ---
        roiL = crop(img, cfg["left"])
        roiR = crop(img, cfg["right"])

        L_closed, L_diff = lid_is_closed_by_template(roiL, refL, corr_thr=0.4)
        R_closed, R_diff = lid_is_closed_by_template(roiR, refR, corr_thr=0.4)

        L_open = not L_closed
        R_open = not R_closed

        lid_state = state_from_lr(L_open, R_open)

        # --- 2) 若 open 才做「上蓋黏餅」檢測 ---
        stuck_L = False
        stuck_R = False
        L_brown = L_v = None
        R_brown = R_v = None

        if L_open:
            roi_ul = crop(img, waffle_cfg["upper_left"])
            L_brown, L_v = waffle_stuck_score(roi_ul)
            stuck_L = decide_stuck(L_brown, L_v, brown_thr=BROWN_THR, v_thr=V_THR)

        if R_open:
            roi_ur = crop(img, waffle_cfg["upper_right"])
            R_brown, R_v = waffle_stuck_score(roi_ur)
            stuck_R = decide_stuck(R_brown, R_v, brown_thr=BROWN_THR, v_thr=V_THR)

        # --- 3) 下方烤盤：有無鬆餅 + 中心位置（只有在 lid open 且沒有上蓋黏餅才檢查） ---
        bottom_has_L = bottom_has_R = False
        cL = cR = (None, None)
        ratioL = ratioR = 0.0
        maskL = maskR = None

        BOTTOM_MIN_RATIO = 0.08   # 非黑比例多少算「有鬆餅/鬆餅液」
        V_DARK_THR = 85           # 黑色判斷門檻(跟你 black_ratio 一致即可)

        if L_open and (not stuck_L):
            # 注意：下烤盤 ROI 就用 cfg["left"] 這個 ROI（open 時它看到的就是下烤盤內容）
            bottom_has_L, cL, ratioL, maskL = waffle_bottom_center(roiL, v_dark_thr=V_DARK_THR, min_ratio=BOTTOM_MIN_RATIO)

        if R_open and (not stuck_R):
            bottom_has_R, cR, ratioR, maskR = waffle_bottom_center(roiR, v_dark_thr=V_DARK_THR, min_ratio=BOTTOM_MIN_RATIO)


        # --- 視覺化 ---
        vis = img.copy()

        # ---- 準備回傳代碼 ----
        # ---- 1) 左右 side code ----
        L_code = make_side_code(L_open, stuck_L, bottom_has_L, 'L')
        R_code = make_side_code(R_open, stuck_R, bottom_has_R, 'R')

        # dx,dy 只在「下方有鬆餅」才有意義；其他狀態一律輸出 0,0
        if L_code != 3:  # 左邊下方有鬆餅才是 code=3
            L_dx, L_dy = 0, 0
        if R_code != 7:  # 右邊下方有鬆餅才是 code=7
            R_dx, R_dy = 0, 0

        L_dx_o = round(float(L_dx), 1)
        L_dy_o = round(float(L_dy), 1)
        R_dx_o = round(float(R_dx), 1)
        R_dy_o = round(float(R_dy), 1)

        msg_tuple = (L_code, L_dx_o, L_dy_o, R_code, R_dx_o, R_dy_o)
        #msg_tuple = (L_code, int(L_dx), int(L_dy), R_code, int(R_dx), int(R_dy))
        msg_str = f"{msg_tuple}"  # 例如 "(4, 0, 0, 7, 1, -2)"

        # ---- 2) 顯示在畫面上（左上角）----
        # vis 是你最後拿來 imshow 的那張圖（如果你直接畫在 frame 上也可以用 frame）
        x, y = 20, 70
        cv2.putText(vis, f"CODE: {msg_str}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        # （可選）也可以把每邊各自顯示
        cv2.putText(vis, f"L={L_code} dxdy=({L_dx},{L_dy})", (x, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"R={R_code} dxdy=({R_dx},{R_dy})", (x, y+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


        # 下蓋 ROI（open/close 判斷用）
        draw_box(vis, cfg["left"],  (0,255,0) if L_open else (0,0,255),
                f"L-> {'OPEN' if L_open else 'CLOSE'} L_diff = {L_diff:.1f}")

        draw_box(vis, cfg["right"], (0,255,0) if R_open else (0,0,255),
                f"R-> {'OPEN' if R_open else 'CLOSE'} R_diff = {R_diff:.1f}")


        # 上蓋 ROI（黏餅判斷用）—只有 open 才顯示狀態
        if L_open:
            #txt = f"STUCK-L {'YES' if stuck_L else 'NO '} brown={L_brown:.3f} v={R_v:.1f}"
            txt = f"STUCK-L {'YES' if stuck_L else 'NO '}"
            draw_box(vis, waffle_cfg["upper_left"], (0,0,255) if stuck_L else (0,255,0), txt)
        else:
            draw_box(vis, waffle_cfg["upper_left"], (128,128,128), "STUCK-L (skip, lid closed)")

        if R_open:
            #txt = f"STUCK-R {'YES' if stuck_R else 'NO '} brown={R_brown:.3f} v={R_v:.1f}"
            txt = f"STUCK-R {'YES' if stuck_R else 'NO '}"
            draw_box(vis, waffle_cfg["upper_right"], (0,0,255) if stuck_R else (0,255,0), txt)
        else:
            draw_box(vis, waffle_cfg["upper_right"], (128,128,128), "STUCK-R (skip, lid closed)")

        # 警示
        alert = []
        if stuck_L: alert.append("LEFT_LID_WAFFLE_STUCK")
        if stuck_R: alert.append("RIGHT_LID_WAFFLE_STUCK")

        cv2.putText(vis, f"STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 6)
        cv2.putText(vis, f"STATE: {lid_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if alert:
            msg = "ALERT: " + " | ".join(alert)
            cv2.putText(vis, msg, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 6)
            cv2.putText(vis, msg, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        #cv2.putText(vis, f"{os.path.basename(p)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
        #            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
        #cv2.putText(vis, f"{os.path.basename(p)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
        #            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # --- 4) 決策輸出：ready / center offset ---
        ready = False
        center_msgs = []
        #l_code_msg = []
        #r_code_msg = []

        # ready：當「有開蓋」且「開的那一側」下烤盤都沒有鬆餅/鬆餅液
        #（輸出1~8的code）
        if lid_state != "ALL_CLOSED" and (not alert):
            # 只檢查 open 的那側
            okL = (not L_open) or (not bottom_has_L)
            okR = (not R_open) or (not bottom_has_R)
            if okL and okR:
                ready = True
                #l_code_msg.append(f"(1, 0, 0)")
                #l_code_msg.append(f"(5, 0, 0)")

        # center 訊息（只對「open 且下烤盤有鬆餅」的那側算）
        if bottom_has_L and cL[0] is not None:
            # ROI座標 -> 全圖座標
            x,y,w,h = cfg["left"]
            gx, gy = x + cL[0], y + cL[1]
            dx, dy = (cL[0] - w/2.0), (cL[1] - h/2.0)  # 以ROI中心為0的偏移（像素）
            center_msgs.append(f"L_center=({gx:.1f},{gy:.1f}) d=({dx:.1f},{dy:.1f}) ratio={ratioL:.2f}")
            #l_code_msg.append(f"(3, {dx:.1f},{dy:.1f})")

        if bottom_has_R and cR[0] is not None:
            x,y,w,h = cfg["right"]
            gx, gy = x + cR[0], y + cR[1]
            dx, dy = (cR[0] - w/2.0), (cR[1] - h/2.0)
            center_msgs.append(f"R_center=({gx:.1f},{gy:.1f}) d=({dx:.1f},{dy:.1f}) ratio={ratioR:.2f}")
            #r_code_msg.append(f"(7, {dx:.1f},{dy:.1f})")

        if ready:
            cv2.putText(vis, "READY: no waffle on bottom", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 6)
            cv2.putText(vis, "READY: no waffle on bottom", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if center_msgs:
            msg = " | ".join(center_msgs)
            cv2.putText(vis, msg, (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
            cv2.putText(vis, msg, (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if bottom_has_L and cL[0] is not None:
            x,y,w,h = cfg["left"]
            cv2.circle(vis, (int(x + cL[0]), int(y + cL[1])), 6, (0,255,255), -1)

        if bottom_has_R and cR[0] is not None:
            x,y,w,h = cfg["right"]
            cv2.circle(vis, (int(x + cR[0]), int(y + cR[1])), 6, (0,255,255), -1)

        cv2.imshow("Camera Infer", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("debug_camera.jpg", vis)
            print("[OK] Saved debug_camera.jpg")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
