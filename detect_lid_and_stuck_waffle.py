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
    BROWN_THR = 0.3
    V_THR = 105

    paths = list_images(INFER_FOLDER)
    if not paths: raise RuntimeError("No images in infer folder")

    idx = 0
    print("[INFO] Keys: space=next, b=back, s=save, q=quit")

    while True:
        idx = max(0, min(idx, len(paths)-1))
        p = paths[idx]
        img = cv2.imread(p)
        if img is None:
            idx += 1
            if idx >= len(paths): break
            continue

        # --- 1) 判斷 lids open/close（左右各自） ---
        L_black = compute_black_ratio(crop(img, cfg["left"]))
        R_black = compute_black_ratio(crop(img, cfg["right"]))
        L_open = (L_black > black_thr)
        R_open = (R_black > black_thr)
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

        # --- 視覺化 ---
        vis = img.copy()

        # 下蓋 ROI（open/close 判斷用）
        draw_box(vis, cfg["left"],  (0,255,0) if L_open else (0,0,255), f"L->{'OPEN' if L_open else 'CLOSE'}")
        draw_box(vis, cfg["right"], (0,255,0) if R_open else (0,0,255), f"R->{'OPEN' if R_open else 'CLOSE'}")

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

        cv2.putText(vis, f"{os.path.basename(p)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
        cv2.putText(vis, f"{os.path.basename(p)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Infer Lid + Stuck Waffle", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            idx += 1
            if idx >= len(paths): break
        elif key == ord('b'):
            idx -= 1
        elif key == ord('s'):
            out = f"debug_{idx:05d}.jpg"
            cv2.imwrite(out, vis)
            print(f"[OK] Saved {out}")

    cv2.destroyAllWindows()
