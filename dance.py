import cv2
import mediapipe as mp
import numpy as np
import csv  # 新增：匯出用

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
pose_connections = mp_pose.POSE_CONNECTIONS

# 影片與攝影機
cap_video = cv2.VideoCapture('《青春有你2》主題曲《YES！OK！》舞台onetake直拍——劉雨昕｜愛奇藝台灣站.mp4')
cap_cam = cv2.VideoCapture(0)

# 嘗試讀取影片第一幀
success, first_frame = cap_video.read()
if not success:
    print("❌ 無法讀取影片。")
    cap_video.release()
    exit()

# 縮放設定
scale = 0.5
frame_height, frame_width = first_frame.shape[:2]
target_size = (int(frame_width * scale), int(frame_height * scale))

# 評分紀錄陣列（新）
score_list = []

# 評分函數
def calculate_pose_similarity(landmarks1, landmarks2, w, h):
    if not landmarks1 or not landmarks2:
        return 0.0
    total_dist = 0
    count = 0
    for i in range(len(landmarks1)):
        x1, y1 = landmarks1[i].x * w, landmarks1[i].y * h
        x2, y2 = landmarks2[i].x * w, landmarks2[i].y * h
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        total_dist += dist
        count += 1
    avg_dist = total_dist / count if count > 0 else 1e9
    score = max(0, 100 - avg_dist * 0.2)
    return round(score, 1)

# 主迴圈
frame_index = 0  # 幀數計數器
while cap_video.isOpened() and cap_cam.isOpened():
    ret_vid, frame_vid = cap_video.read()
    ret_cam, frame_cam = cap_cam.read()
    if not ret_vid or not ret_cam:
        print("✅ 播放完畢或攝影機異常")
        break

    # 縮小畫面
    frame_vid = cv2.resize(frame_vid, target_size)
    frame_cam = cv2.resize(frame_cam, target_size)
    h, w = frame_vid.shape[:2]

    # 建立黑底畫布
    bkb_vid = np.zeros_like(frame_vid)
    bkb_cam = np.zeros_like(frame_cam)

    # 處理影片骨架
    result_vid = pose.process(cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB))
    if result_vid.pose_landmarks:
        mp_drawing.draw_landmarks(frame_vid, result_vid.pose_landmarks, pose_connections, drawing_spec)
        mp_drawing.draw_landmarks(bkb_vid, result_vid.pose_landmarks, pose_connections, drawing_spec)

    # 處理攝影機骨架
    result_cam = pose.process(cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB))
    if result_cam.pose_landmarks:
        mp_drawing.draw_landmarks(frame_cam, result_cam.pose_landmarks, pose_connections, drawing_spec)
        mp_drawing.draw_landmarks(bkb_cam, result_cam.pose_landmarks, pose_connections, drawing_spec)

    # 計算評分
    score = 0
    if result_vid.pose_landmarks and result_cam.pose_landmarks:
        score = calculate_pose_similarity(
            result_vid.pose_landmarks.landmark,
            result_cam.pose_landmarks.landmark,
            w, h
        )
        score_list.append((frame_index, score))  # ➕ 紀錄到列表中

    # 合成畫面
    top_row = np.hstack((frame_vid, frame_cam))
    bottom_row = np.hstack((bkb_vid, bkb_cam))
    final_frame = np.vstack((top_row, bottom_row))

    # 顯示評分
    cv2.putText(final_frame, f'Score: {score}/100', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('期末報告：骨架評分系統', final_frame)

    frame_index += 1  # 幀數 +1
    if cv2.waitKey(1) & 0xFF == 27:
        print("🔚 手動結束。")
        break

# 釋放資源
cap_video.release()
cap_cam.release()
cv2.destroyAllWindows()

# === 匯出分數資料到 CSV ===
FPS = 30  # 如果你的影片是 30fps，這裡設定 30

with open('scores1.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Time(sec)', 'Score'])
    for frame, score in score_list:
        time_sec = round(frame / FPS, 2)
        writer.writerow([frame, time_sec, score])

print("✅ 分數已儲存為 scores1.csv")

import csv

lowest_score = float('inf')
lowest_frame = 0
lowest_time = 0

FPS = 30  # 如果你的影片是 30 FPS

with open('scores1.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        score = float(row['Score'])
        frame = int(row['Frame'])
        time = float(row['Time(sec)'])
        if score < lowest_score:
            lowest_score = score
            lowest_frame = frame
            lowest_time = time

print(f"最低分數：{lowest_score}，Frame：{lowest_frame}，時間：約 {lowest_time:.2f} 秒")

import cv2

# 影片載入
video_path = '《青春有你2》主題曲《YES！OK！》舞台onetake直拍——劉雨昕｜愛奇藝台灣站.mp4'
cap = cv2.VideoCapture(video_path)

# 跳到最低分的 frame
cap.set(cv2.CAP_PROP_POS_FRAMES, lowest_frame)

# 讀取該幀
ret, frame = cap.read()
if ret:
    # 顯示畫面
    cv2.imshow('最低分數畫面', frame)
    # 儲存截圖
    cv2.imwrite('worst_frame.jpg', frame)
    print("✅ 已儲存畫面為 worst_frame.jpg")
    cv2.waitKey(0)
else:
    print("❌ 無法讀取該幀")



cap.release()
cv2.destroyAllWindows()
