import cv2
import json
import os
import glob
import numpy as np

ROI_PATH = "roi_config.json"
THR_PATH = "thresholds.json"

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    paths=[]
    for e in exts:
        paths += glob.glob(os.path.join(folder, e))
    paths.sort()
    return paths

def select_rois(frame):
    print("[INFO] Select LEFT lid ROI, ENTER confirm")
    x,y,w,h = cv2.selectROI("Select LEFT ROI", frame, True, False)
    cv2.destroyWindow("Select LEFT ROI")
    if w==0 or h==0: raise RuntimeError("Left ROI canceled")

    print("[INFO] Select RIGHT lid ROI, ENTER confirm")
    x2,y2,w2,h2 = cv2.selectROI("Select RIGHT ROI", frame, True, False)
    cv2.destroyWindow("Select RIGHT ROI")
    if w2==0 or h2==0: raise RuntimeError("Right ROI canceled")

    cfg={"left":[int(x),int(y),int(w),int(h)],
         "right":[int(x2),int(y2),int(w2),int(h2)]}
    with open(ROI_PATH,"w",encoding="utf-8") as f:
        json.dump(cfg,f,ensure_ascii=False,indent=2)
    print(f"[OK] saved {ROI_PATH}")
    return cfg

def crop(img, roi):
    x,y,w,h = roi
    return img[y:y+h, x:x+w].copy()

def compute_features(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:,:,2], (5,5), 0)
    v_dark_thr = 85
    black_ratio = float((v < v_dark_thr).mean())

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float((edges > 0).mean())

    return black_ratio, edge_density

def auto_calibrate_thresholds(step1_folder, cfg):
    """
    假設 step1_folder 內含 10 張影像，且你可以用鍵盤標註每張是 CLOSED / OPEN
    最快：按 'c' 標註全關、按 'o' 標註全開、space 下一張、q 結束
    最後以 closed/open 的平均值取中點當 threshold。
    """
    paths = list_images(step1_folder)
    if not paths:
        raise RuntimeError("No images in step1 folder")

    closed_feats=[]
    open_feats=[]

    idx=0
    print("\n[CAL] Label step1 images: c=CLOSED, o=OPEN, space=skip/next, q=finish")
    while True:
        if idx>=len(paths): break
        img = cv2.imread(paths[idx])
        if img is None:
            idx += 1
            continue

        L = compute_features(crop(img, cfg["left"]))
        R = compute_features(crop(img, cfg["right"]))
        # pooled
        b = (L[0] + R[0]) / 2.0
        e = (L[1] + R[1]) / 2.0

        vis = img.copy()
        cv2.putText(vis, f"{os.path.basename(paths[idx])} [{idx+1}/{len(paths)}]  b={b:.3f} e={e:.3f}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 6)
        cv2.putText(vis, f"{os.path.basename(paths[idx])} [{idx+1}/{len(paths)}]  b={b:.3f} e={e:.3f}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, "Press c=CLOSED, o=OPEN, space=next, q=finish",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 6)
        cv2.putText(vis, "Press c=CLOSED, o=OPEN, space=next, q=finish",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Step1-Label", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            closed_feats.append((b,e))
            idx += 1
        elif key == ord('o'):
            open_feats.append((b,e))
            idx += 1
        elif key == ord(' '):
            idx += 1
        else:
            idx += 1

    cv2.destroyWindow("Step1-Label")

    if len(closed_feats) < 1 or len(open_feats) < 1:
        print("[WARN] Not enough labels for calibration. thresholds will be fallback.")
        thr = {"black_thr": 0.28, "edge_thr": 0.07, "calibrated": False}
    else:
        cb = float(np.mean([x[0] for x in closed_feats]))
        ce = float(np.mean([x[1] for x in closed_feats]))
        ob = float(np.mean([x[0] for x in open_feats]))
        oe = float(np.mean([x[1] for x in open_feats]))
        thr = {
            "black_thr": float((cb + ob)/2.0),
            "edge_thr":  float((ce + oe)/2.0),
            "calibrated": True,
            "stats": {
                "closed_mean": {"black": cb, "edge": ce, "n": len(closed_feats)},
                "open_mean":   {"black": ob, "edge": oe, "n": len(open_feats)}
            }
        }

    with open(THR_PATH, "w", encoding="utf-8") as f:
        json.dump(thr, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved {THR_PATH}: {thr}")
    return thr

if __name__ == "__main__":
    STEP1_FOLDER = r"./select_roi"  

    paths = list_images(STEP1_FOLDER)
    if not paths:
        raise RuntimeError("No images in step1 folder")

    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError("Cannot read first image in step1 folder")

    cfg = select_rois(first)
    auto_calibrate_thresholds(STEP1_FOLDER, cfg)
    cv2.destroyAllWindows()
