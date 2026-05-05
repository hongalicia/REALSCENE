import cv2
import os

def extract_frames(video_path, output_folder="frames_output", frame_skip=1):
    """
    將影片拆分成圖片幀並儲存到指定資料夾。

    參數:
    video_path (str): 影片檔案的路徑 (例如: 'my_video.mp4')
    output_folder (str): 儲存圖片幀的資料夾名稱
    frame_skip (int): 每隔多少幀取一張圖片 (1表示取每一幀)
    """
    
    # 檢查並創建輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 創建輸出資料夾: {output_folder}")

    # 打開影片檔案
    cap = cv2.VideoCapture(video_path)
    
    # 檢查影片是否成功開啟
    if not cap.isOpened():
        print(f"❌ 錯誤：無法開啟影片檔案 {video_path}")
        return

    frame_count = 0  # 影片幀的計數器
    saved_count = 0  # 成功儲存的圖片計數器

    print(f"🚀 開始處理影片: {video_path}...")
    
    while True:
        # 讀取下一幀
        ret, frame = cap.read()
        
        # ret 為 True 表示成功讀取；讀取失敗或影片結束時 ret 為 False
        if not ret:
            break

        # 根據 frame_skip 參數決定是否儲存這一幀
        if frame_count % frame_skip == 0:
            # 建立圖片檔名 (例如: frame_00001.jpg, frame_00002.jpg)
            image_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            
            # 儲存圖片幀
            cv2.imwrite(image_filename, frame)
            saved_count += 1
            
            # 顯示進度
            if saved_count % 100 == 0:
                 print(f"   已儲存 {saved_count} 張圖片...")

        frame_count += 1

    # 釋放影片捕捉物件
    cap.release()
    
    print("-" * 30)
    print(f"🎉 處理完成！")
    print(f"總共處理了 {frame_count} 幀影片。")
    print(f"總共儲存了 {saved_count} 張圖片到資料夾：{output_folder}")

# --- 設定您的影片路徑和參數 ---
# 請將 'input_video.mp4' 替換為您的影片檔案路徑
VIDEO_FILE_PATH = '1211.mp4' 
# 設定輸出資料夾名稱
OUTPUT_DIR = 'video_frames_output'
# 設定抽幀間隔 (1: 每一幀都存, 10: 每隔 10 幀存一張，可加快處理速度和減少圖片數量)
SKIP_FRAMES = 10 

# 執行函式
extract_frames(
    video_path=VIDEO_FILE_PATH, 
    output_folder=OUTPUT_DIR, 
    frame_skip=SKIP_FRAMES
)