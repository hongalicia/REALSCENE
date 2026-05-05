import cv2
import json
import os
import glob

WAFFLE_ROI_PATH = "waffle_roi_config.json"

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    paths=[]
    for e in exts:
        paths += glob.glob(os.path.join(folder,e))
    paths.sort()
    return paths

def select_waffle_rois(frame):
    print("[INFO] Select LEFT UPPER-LID ROI (where waffle may stick), ENTER confirm.")
    x, y, w, h = cv2.selectROI("Select LEFT UPPER-LID ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select LEFT UPPER-LID ROI")
    if w == 0 or h == 0:
        raise RuntimeError("Left upper-lid ROI canceled.")

    print("[INFO] Select RIGHT UPPER-LID ROI (where waffle may stick), ENTER confirm.")
    x2, y2, w2, h2 = cv2.selectROI("Select RIGHT UPPER-LID ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select RIGHT UPPER-LID ROI")
    if w2 == 0 or h2 == 0:
        raise RuntimeError("Right upper-lid ROI canceled.")

    cfg = {
        "upper_left": [int(x), int(y), int(w), int(h)],
        "upper_right": [int(x2), int(y2), int(w2), int(h2)]
    }
    with open(WAFFLE_ROI_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {WAFFLE_ROI_PATH}")
    return cfg

if __name__ == "__main__":
    # 用「最容易看清楚上蓋內側」的圖片來框 ROI（例如你提供的黏餅圖）
    SETUP_FOLDER = r"./folderWaffleROI"

    paths = list_images(SETUP_FOLDER)
    if not paths:
        raise RuntimeError(f"No images found in {SETUP_FOLDER}")

    img = cv2.imread(paths[0])
    if img is None:
        raise RuntimeError(f"Cannot read {paths[0]}")

    select_waffle_rois(img)
    cv2.destroyAllWindows()
