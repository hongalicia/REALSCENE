import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# 先用 1920x1080 @ 30 之類保守值
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# 可加：指定 MJPG（常讓 BRIO 比較好談）
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

ret, frame = cap.read()
print("read:", ret, "shape:", None if frame is None else frame.shape)
cap.release()
