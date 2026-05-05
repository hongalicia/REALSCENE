import cv2
import json
import os
import glob
import numpy as np
from collections import deque

CFG_PATH = "roi_config.json"

# ---------- Utils ----------
def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths.sort()
    return paths

def select_rois(frame):
    print("[INFO] Select LEFT lid ROI, ENTER/SPACE confirm, ESC cancel.")
    x, y, w, h = cv2.selectROI("Select LEFT ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select LEFT ROI")
    if w == 0 or h == 0:
        raise RuntimeError("Left ROI selection canceled.")

    print("[INFO] Select RIGHT lid ROI, ENTER/SPACE confirm, ESC cancel.")
    x2, y2, w2, h2 = cv2.selectROI("Select RIGHT ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select RIGHT ROI")
    if w2 == 0 or h2 == 0:
        raise RuntimeError("Right ROI selection canceled.")

    cfg = {"left": [int(x), int(y), int(w), int(h)],
           "right": [int(x2), int(y2), int(w2), int(h2)]}
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved ROI config to {CFG_PATH}")
    return cfg

def load_rois():
    if not os.path.exists(CFG_PATH):
        return None
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w].copy()

# ---------- Features ----------
def compute_features(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)

    v_dark_thr = 85
    black_ratio = float((v_blur < v_dark_thr).mean())

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 60, 140)
    edge_density = float((edges > 0).mean())

    v_mean = float(v_blur.mean())
    return black_ratio, edge_density, v_mean

# ---------- Calibration ----------
class Calibrator:
    def __init__(self):
        self.closed = []  # pooled (L+R)
        self.open = []
        self.thresholds_ready = False
        self.black_thr = None
        self.edge_thr = None

    def add_closed(self, feats_L, feats_R):
        self.closed.append((feats_L[0], feats_L[1]))
        self.closed.append((feats_R[0], feats_R[1]))
        self._fit()

    def add_open(self, feats_L, feats_R):
        self.open.append((feats_L[0], feats_L[1]))
        self.open.append((feats_R[0], feats_R[1]))
        self._fit()

    def _fit(self):
        if len(self.closed) < 1 or len(self.open) < 1:
            return
        cb = np.mean([x[0] for x in self.closed])
        ce = np.mean([x[1] for x in self.closed])
        ob = np.mean([x[0] for x in self.open])
        oe = np.mean([x[1] for x in self.open])
        self.black_thr = float((cb + ob) / 2.0)
        self.edge_thr  = float((ce + oe) / 2.0)
        self.thresholds_ready = True

    def decide_open(self, feats, mode="or"):
        b, e, _ = feats
        # fallback if not calibrated yet
        bt = self.black_thr if self.black_thr is not None else 0.28
        et = self.edge_thr  if self.edge_thr  is not None else 0.07
        if mode == "and":
            return (b > bt) and (e > et)
        return (b > bt) or (e > et)

def state_from_lr(left_open, right_open):
    if left_open and right_open:
        return "ALL_OPEN"
    if (not left_open) and (not right_open):
        return "ALL_CLOSED"
    if left_open and (not right_open):
        return "LEFT_OPEN"
    return "RIGHT_OPEN"

def occlusion_guard(prev_feats, cur_feats):
    if prev_feats is None:
        return False
    pb, pe, pv = prev_feats
    cb, ce, cv = cur_feats
    if abs(cb - pb) > 0.25: return True
    if abs(ce - pe) > 0.12: return True
    if abs(cv - pv) > 60:   return True
    return False

def draw_overlay(frame, cfg, L, R, L_open, R_open, stable_state, calib: Calibrator):
    (lx, ly, lw, lh) = cfg["left"]
    (rx, ry, rw, rh) = cfg["right"]

    cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0,255,0) if L_open else (0,0,255), 2)
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,0) if R_open else (0,0,255), 2)

    lb, le, lv = L
    rb, re, rv = R
    cv2.putText(frame, f"L b={lb:.3f} e={le:.3f} v={lv:.1f} -> {'OPEN' if L_open else 'CLOSE'}",
                (lx, max(20, ly-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(frame, f"R b={rb:.3f} e={re:.3f} v={rv:.1f} -> {'OPEN' if R_open else 'CLOSE'}",
                (rx, max(20, ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    if calib.thresholds_ready:
        cv2.putText(frame, f"THR black={calib.black_thr:.3f} edge={calib.edge_thr:.3f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 6)
        cv2.putText(frame, f"THR black={calib.black_thr:.3f} edge={calib.edge_thr:.3f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.putText(frame, f"STATE: {stable_state}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 6)
    cv2.putText(frame, f"STATE: {stable_state}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

# ---------- Step 1: Setup with folder A ----------
def step1_setup(setup_folder):
    paths = list_images(setup_folder)
    if not paths:
        raise RuntimeError(f"No images found in setup folder: {setup_folder}")

    img = cv2.imread(paths[0])
    if img is None:
        raise RuntimeError(f"Cannot read image: {paths[0]}")

    cfg = load_rois()
    if cfg is None:
        cfg = select_rois(img)
    else:
        print(f"[OK] Loaded ROI config from {CFG_PATH}")

    calib = Calibrator()

    print("\n[STEP1] Setup mode (folder A).")
    print("Keys: c=add CLOSED sample, o=add OPEN sample, space=next image, q=finish setup")
    print("TIP: 如果 folder A 裡有『全關』與『全開』的照片，建議各挑幾張按 c/o 讓門檻自動校正。\n")

    idx = 0
    while True:
        idx = max(0, min(idx, len(paths)-1))
        img = cv2.imread(paths[idx])
        if img is None:
            idx += 1
            if idx >= len(paths): break
            continue

        L = compute_features(crop(img, cfg["left"]))
        R = compute_features(crop(img, cfg["right"]))

        vis = img.copy()
        # 暫時用 calibration 的 fallback threshold 顯示
        L_open = calib.decide_open(L)
        R_open = calib.decide_open(R)
        draw_overlay(vis, cfg, L, R, L_open, R_open, "SETUP", calib)

        cv2.putText(vis, f"SETUP {idx+1}/{len(paths)}  (space next, c closed, o open, q finish)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
        cv2.putText(vis, f"SETUP {idx+1}/{len(paths)}  (space next, c closed, o open, q finish)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Step1-Setup", vis)
        key = cv2.waitKey(0) & 0xFF  # wait for key

        if key == ord('q'):
            break
        elif key == ord(' '):  # space
            idx += 1
            if idx >= len(paths):
                break
        elif key == ord('c'):
            calib.add_closed(L, R)
            print(f"[CAL] add CLOSED. closed={len(calib.closed)} open={len(calib.open)} ready={calib.thresholds_ready}")
        elif key == ord('o'):
            calib.add_open(L, R)
            print(f"[CAL] add OPEN. closed={len(calib.closed)} open={len(calib.open)} ready={calib.thresholds_ready}")
        elif key == ord('r'):
            cfg = select_rois(img)  # optional: reselect ROI during setup

    cv2.destroyWindow("Step1-Setup")
    return cfg, calib

# ---------- Step 2: Inference with folder B ----------
def step2_infer(infer_folder, cfg, calib, decision_mode="or", vote_window=5, vote_need=3):
    paths = list_images(infer_folder)
    if not paths:
        raise RuntimeError(f"No images found in infer folder: {infer_folder}")

    print("\n[STEP2] Inference mode (folder B).")
    print("Keys: space=next image, b=back, s=save screenshot, q=quit")
    print("Note: 會用投票讓狀態更穩（單張判斷仍會顯示在 ROI 上）\n")

    buf = deque(maxlen=vote_window)
    stable_state = "UNKNOWN"
    prevL = None
    prevR = None

    idx = 0
    while True:
        idx = max(0, min(idx, len(paths)-1))
        img_path = paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            if idx >= len(paths):
                break
            continue

        L = compute_features(crop(img, cfg["left"]))
        R = compute_features(crop(img, cfg["right"]))

        guardL = occlusion_guard(prevL, L)
        guardR = occlusion_guard(prevR, R)

        # if guarded, keep previous stable side decision
        if not guardL:
            L_open = calib.decide_open(L, mode=decision_mode)
        else:
            L_open = (stable_state in ["ALL_OPEN", "LEFT_OPEN"])

        if not guardR:
            R_open = calib.decide_open(R, mode=decision_mode)
        else:
            R_open = (stable_state in ["ALL_OPEN", "RIGHT_OPEN"])

        state = state_from_lr(L_open, R_open)
        buf.append(state)
        if len(buf) == vote_window:
            vals, counts = np.unique(list(buf), return_counts=True)
            winner = vals[np.argmax(counts)]
            if counts.max() >= vote_need:
                stable_state = winner

        prevL, prevR = L, R

        vis = img.copy()
        draw_overlay(vis, cfg, L, R, L_open, R_open, stable_state, calib)
        cv2.putText(vis, f"{os.path.basename(img_path)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 6)
        cv2.putText(vis, f"{os.path.basename(img_path)}  {idx+1}/{len(paths)}  (space next, b back, q quit)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Step2-Infer", vis)
        key = cv2.waitKey(0) & 0xFF  # wait for key (manual stepping)

        if key == ord('q'):
            break
        elif key == ord(' '):  # next
            idx += 1
            if idx >= len(paths):
                break
        elif key == ord('b'):  # back
            idx -= 1
        elif key == ord('s'):
            out = f"debug_{idx:05d}.jpg"
            cv2.imwrite(out, vis)
            print(f"[OK] Saved {out}")

    cv2.destroyWindow("Step2-Infer")

# ---------- Main ----------
if __name__ == "__main__":
    # 改成你的資料夾路徑
    SETUP_FOLDER = r"./select_roi"
    INFER_FOLDER = r"./infer_test"

    cfg, calib = step1_setup(SETUP_FOLDER)
    step2_infer(INFER_FOLDER, cfg, calib, decision_mode="or", vote_window=5, vote_need=3)
    cv2.destroyAllWindows()
