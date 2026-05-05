import cv2
import json
import os
import glob
import numpy as np
from collections import deque

ROI_PATH = "roi_config.json"              # detect lid open/close ROI
THR_PATH = "thresholds.json"              # detect black_thr
WAFFLE_ROI_PATH = "waffle_roi_config.json"

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
    """
    很快可用的規則：
    - brown_ratio 過高 => 幾乎一定有鬆餅
    - 或者 mean_v 很高也可能代表有鬆餅（輔助）
    你可以先只用 brown_ratio，必要時再加 mean_v。
    """
    return (brown_ratio > brown_thr) or (mean_v > v_thr)

def draw_box(img, roi, color, text):
    x,y,w,h = roi
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(img, text, (x, max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, text, (x, max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

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
    #BROWN_THR = 0.3
    V_THR = 105

    cap = cv2.VideoCapture(0)  # 或你的相機 index / rtsp url
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    TARGET_W, TARGET_H = 1920, 1080
    BROWN_THR = 0.30  # 你調到有效的門檻

    print("[INFO] Keys: q=quit, s=save screenshot")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # 1) resize to match ROI coordinate system
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        # 2) lid open/close (black_ratio)
        L_black = compute_black_ratio(crop(img, cfg["left"]))
        R_black = compute_black_ratio(crop(img, cfg["right"]))
        L_open = (L_black > black_thr)
        R_open = (R_black > black_thr)
        lid_state = state_from_lr(L_open, R_open)

        # 3) stuck waffle detection only when open
        stuck_L = stuck_R = False
        if L_open:
            L_brown, L_v = waffle_stuck_score(crop(img, waffle_cfg["upper_left"]))
            stuck_L = (L_brown > BROWN_THR)
        if R_open:
            R_brown, R_v = waffle_stuck_score(crop(img, waffle_cfg["upper_right"]))
            stuck_R = (R_brown > BROWN_THR)

        # 4) visualize / alert (沿用你原本 draw_box / overlay)
        vis = img.copy()
        # ... (照你原本那套畫框與文字)
        cv2.imshow("Camera Infer", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("debug_camera.jpg", vis)
            print("[OK] Saved debug_camera.jpg")

    cap.release()
    cv2.destroyAllWindows()    
