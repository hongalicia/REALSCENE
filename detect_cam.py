import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 先用 DirectShow
    ok = cap.isOpened()
    print(i, "OK" if ok else "FAIL")
    cap.release()
