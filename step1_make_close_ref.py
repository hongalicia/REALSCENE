import cv2, os, glob, json
import numpy as np

ROI_PATH = "roi_config.json"
OUT_NPZ  = "lid_close_refs.npz"

def list_images(folder):
    exts=("*.jpg","*.jpeg","*.png","*.bmp")
    paths=[]
    for e in exts:
        paths += glob.glob(os.path.join(folder,e))
    paths.sort()
    return paths

def crop(img, roi):
    x,y,w,h = roi
    return img[y:y+h, x:x+w].copy()

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    STEP1_FOLDER = r"./select_roi"   # 你的 step1 10 張圖資料夾

    cfg = json.load(open(ROI_PATH,"r",encoding="utf-8"))

    paths = list_images(STEP1_FOLDER)
    if not paths:
        raise RuntimeError("No images in step1 folder")

    closed_L = []
    closed_R = []

    print("[INFO] Step1 labeling: c=closed, o=open, space=next, q=finish")
    idx=0
    while idx < len(paths):
        img = cv2.imread(paths[idx])
        if img is None:
            idx += 1
            continue

        vis = img.copy()
        cv2.putText(vis, f"{os.path.basename(paths[idx])} {idx+1}/{len(paths)}  (c closed / o open / space next / q finish)",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 6)
        cv2.putText(vis, f"{os.path.basename(paths[idx])} {idx+1}/{len(paths)}  (c closed / o open / space next / q finish)",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Step1 Label", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            closed_L.append(to_gray(crop(img, cfg["left"])))
            closed_R.append(to_gray(crop(img, cfg["right"])))
            print("[TAG] CLOSED:", os.path.basename(paths[idx]))
            idx += 1
        elif k == ord('o') or k == ord(' '):
            idx += 1
        else:
            idx += 1

    cv2.destroyAllWindows()

    if len(closed_L) < 2 or len(closed_R) < 2:
        raise RuntimeError("Need at least 2 CLOSED samples for each side. Please tag more images with 'c'.")

    # 用 median 當 close reference（抗雜訊、抗小遮擋）
    refL = np.median(np.stack(closed_L, axis=0), axis=0).astype(np.uint8)
    refR = np.median(np.stack(closed_R, axis=0), axis=0).astype(np.uint8)

    # 用標註的 CLOSED 樣本去估 diff 分布，設定門檻
    diffsL = [float(np.mean(cv2.absdiff(x, refL))) for x in closed_L]
    diffsR = [float(np.mean(cv2.absdiff(x, refR))) for x in closed_R]

    # 門檻：取「closed diff 的高位數」再加一點 buffer
    thrL = float(np.percentile(diffsL, 95) + 2.0)
    thrR = float(np.percentile(diffsR, 95) + 2.0)

    np.savez_compressed(OUT_NPZ, refL=refL, refR=refR, thrL=thrL, thrR=thrR)
    cv2.imwrite("close_ref_left.png", refL)
    cv2.imwrite("close_ref_right.png", refR)

    print("[OK] Saved:", OUT_NPZ)
    print("     thrL=", thrL, "thrR=", thrR)
