import cv2
import time

def test_camera():
    # 根據你的 Log，相機 index 是 1，backend 是 DSHOW
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"無法開啟相機 index {cam_index}")
        return

    # 設定解析度與格式 (依據你原本的設定)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("相機已開啟，按下 'q' 鍵退出測試")
    
    # 建立視窗並強制置頂
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("讀取影像失敗...")
                break

            # 檢查影像平均亮度，如果是 0 代表全黑
            mean_brightness = frame.mean()
            
            # 在畫面上印出資訊
            cv2.putText(frame, f"Brightness: {mean_brightness:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Camera Test", frame)

            # 等待 1ms 處理視窗事件，按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("測試結束")

if __name__ == "__main__":
    test_camera()