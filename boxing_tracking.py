import cv2
import mediapipe as mp
import pyautogui
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1
)

# Open camera (Webcam)
cap = cv2.VideoCapture(0)

print("Program Boxing Responsif & Anti-Error Started! Press 'q' for leave.")

left_is_punching = False
right_is_punching = False
last_punch_time = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membuka kamera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Konversi format warna BGR ke RGB untuk MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Deteksi pose tubuh
        results = landmarker.detect(mp_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]

        
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Ambil titik penting: Tangan Kiri (15), Tangan Kanan (16), Bahu Kiri (11), Bahu Kanan (12), Hidung (0)
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            nose = landmarks[0]

            center_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            current_time = time.time()

            # Hitung dorongan relatif terhadap bahu (sumbu Z)
            left_push = left_wrist.z - left_shoulder.z
            right_push = right_wrist.z - right_shoulder.z

            if current_time - last_punch_time > 0.25:
                
                
                if left_push < -0.18:
                    if not left_is_punching:
                        print("LIGHT PUNCH (Tangan Kiri) ")
                        pyautogui.click()
                        left_is_punching = True
                        last_punch_time = current_time
                elif left_push > -0.08:
                    left_is_punching = False 

                
                if right_push < -0.22:
                    if not right_is_punching:
                        print("HEAVY PUNCH (Tangan Kanan) ")
                        pyautogui.click()
                        right_is_punching = True
                        last_punch_time = current_time
                elif right_push > -0.10:
                    right_is_punching = False  # Reset saat tangan ditarik mundur

            if nose.x < center_shoulder_x - 0.12:
                pyautogui.press('a')
            elif nose.x > center_shoulder_x + 0.12:
                pyautogui.press('d')

        # Tampilkan jendela kamera dengan titik tracking hijau
        cv2.imshow('Roblox Boxing - Tasks API Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
