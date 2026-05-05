import cv2
import json
import os
import glob
import numpy as np
from collections import deque

ROI_PATH = "roi_config.json"
THR_PATH = "thresholds.json"

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

def compute_features(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:,:,2], (5,5), 0)
    v_dark_thr=85
    black_ratio=float((v < v_dark_thr).mean())

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 140)
    edge_density=float((edges>0).mean())
    v_mean=float(v.mean())
    return black_ratio, edge_density, v_mean

def occlusion_guard(prev_feats, cur_feats):
    if prev_feats is None: return False
    pb,pe,pv = prev_feats
    cb,ce,cv = cur_feats
    if abs(cb-pb) > 0.25: return True
    if abs(ce-pe) > 0.12: return True
    if abs(cv-pv) > 60:   return True
    return False

def decide_open(feats, black_thr, edge_thr, mode="or"):
    b, e, _ = feats
    # 開：黑多 + 邊緣少
    #return (b > black_thr) and (e < edge_thr)
    return (b > black_thr)

def state_from_lr(L_open, R_open):
    if L_open and R_open: return "ALL_OPEN"
    if (not L_open) and (not R_open): return "ALL_CLOSED"
    if L_open and (not R_open): return "LEFT_OPEN"
    return "RIGHT_OPEN"

def draw_overlay(img, cfg, L, R, L_open, R_open, stable_state, black_thr, edge_thr, path_text):
    (lx,ly,lw,lh)=cfg["left"]
    (rx,ry,rw,rh)=cfg["right"]
    cv2.rectangle(img, (lx,ly), (lx+lw,ly+lh), (0,255,0) if L_open else (0,0,255), 2)
    cv2.rectangle(img, (rx,ry), (rx+rw,ry+rh), (0,255,0) if R_open else (0,0,255), 2)

    lb,le,lv=L
    rb,re,rv=R
    #cv2.putText(img, f"L b={lb:.3f} e={le:.3f} v={lv:.1f} -> {'OPEN' if L_open else 'CLOSE'}",
    #            (lx, max(20,ly-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    #cv2.putText(img, f"R b={rb:.3f} e={re:.3f} v={rv:.1f} -> {'OPEN' if R_open else 'CLOSE'}",
    #            (rx, max(20,ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(img, f"L -> {'OPEN' if L_open else 'CLOSE'}",
                (lx, max(20,ly-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(img, f"R -> {'OPEN' if R_open else 'CLOSE'}",
                (rx, max(20,ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.putText(img, f"THR black={black_thr:.3f} edge={edge_thr:.3f}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, f"THR black={black_thr:.3f} edge={edge_thr:.3f}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(img, f"STATE: {stable_state}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 6)
    cv2.putText(img, f"STATE: {stable_state}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(img, path_text, (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
    cv2.putText(img, path_text, (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

if __name__ == "__main__":
    INFER_FOLDER = r"./infer_test"  
    decision_mode="or"
    vote_window=5
    vote_need=3

    if not os.path.exists(ROI_PATH):
        raise RuntimeError("roi_config.json not found. Run roi_labeler.py first.")
    if not os.path.exists(THR_PATH):
        raise RuntimeError("thresholds.json not found. Run roi_labeler.py first.")

    cfg = json.load(open(ROI_PATH,"r",encoding="utf-8"))
    thr = json.load(open(THR_PATH,"r",encoding="utf-8"))
    black_thr = float(thr.get("black_thr", 0.28))
    edge_thr  = float(thr.get("edge_thr", 0.07))

    paths = list_images(INFER_FOLDER)
    if not paths:
        raise RuntimeError("No images in infer folder")

    buf=deque(maxlen=vote_window)
    stable_state="UNKNOWN"
    prevL=None
    prevR=None
    idx=0

    cap = cv2.VideoCapture(0)  # 0=預設相機；或改成你的相機 index / RTSP URL
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("[INFO] Keys: q=quit, s=save screenshot")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # --- 以下這段保留你原本的 ROI 計算邏輯 ---
        L = compute_features(crop(img, cfg["left"]))
        R = compute_features(crop(img, cfg["right"]))

        # 你已決定只用 black_ratio：假設 compute_features 回傳 (black, edge, v_mean)
        L_open = (L[0] > black_thr)
        R_open = (R[0] > black_thr)

        state = state_from_lr(L_open, R_open)

        # (可選) 投票濾波：把 state 放進 buf 後算 stable_state
        # buf.append(state) ...

        vis = img.copy()
        draw_overlay(vis, cfg, L, R, L_open, R_open, state, black_thr, edge_thr, "CAMERA")

        cv2.imshow("Infer-Camera", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("debug_camera.jpg", vis)
            print("[OK] Saved debug_camera.jpg")

    cap.release()
    cv2.destroyAllWindows()

