import cv2
import os
import mediapipe as mp
import numpy as np

# ---------- Overlay Function ----------
def overlay_png(bg, overlay, x, y, size):
    overlay = cv2.resize(overlay, size)
    h, w = overlay.shape[:2]

    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return

    # RGB image
    if overlay.shape[2] == 3:
        bg[y:y+h, x:x+w] = overlay
        return

    # RGBA image
    for i in range(h):
        for j in range(w):
            alpha = overlay[i, j][3] / 255.0
            if alpha > 0:
                bg[y+i, x+j] = (
                    alpha * overlay[i, j][:3] +
                    (1 - alpha) * bg[y+i, x+j]
                )

# ---------- MediaPipe ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dog_nose  = cv2.imread(os.path.join(BASE_DIR, "dog_nose.png"), cv2.IMREAD_UNCHANGED)
left_ear  = cv2.imread(os.path.join(BASE_DIR, "left_ear.png"), cv2.IMREAD_UNCHANGED)
right_ear = cv2.imread(os.path.join(BASE_DIR, "right_ear.png"), cv2.IMREAD_UNCHANGED)
tongue    = cv2.imread(os.path.join(BASE_DIR, "tongue.png"), cv2.IMREAD_UNCHANGED)

if dog_nose is None or left_ear is None or right_ear is None or tongue is None:
    print("❌ One or more images not loaded")
    exit()

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:

            h, w, _ = frame.shape

            # ---------- Eyes ----------
            left_eye  = face.landmark[33]
            right_eye = face.landmark[263]
            eye_dist = int(abs(left_eye.x - right_eye.x) * w)

            # ---------- Nose ----------
            nose = face.landmark[1]
            nx = int(nose.x * w)
            ny = int(nose.y * h)
            nose_size = int(eye_dist * 0.6)

            overlay_png(
                frame,
                dog_nose,
                nx - nose_size // 2,
                ny - nose_size // 2,
                (nose_size, nose_size)
            )

            # ---------- EARS ----------
            ear_size = int(eye_dist * 0.8)

            lx = int(left_eye.x * w)
            ly = int(left_eye.y * h)

            rx = int(right_eye.x * w)
            ry = int(right_eye.y * h)

            ear_y_offset = int(ear_size * 1.3)

            # Left Ear
            overlay_png(
                frame,
                left_ear,
                lx - ear_size,
                ly - ear_y_offset,
                (ear_size, ear_size)
            )

            # Right Ear
            overlay_png(
                frame,
                right_ear,
                rx,
                ry - ear_y_offset,
                (ear_size, ear_size)
            )

            # ---------- TONGUE ----------
            upper_lip = face.landmark[13]
            lower_lip = face.landmark[14]
            left_mouth = face.landmark[78]
            right_mouth = face.landmark[308]

            # Detect mouth open
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            MOUTH_OPEN_THRESHOLD = 15  # adjust if tongue shows too early/late

            if lip_dist > MOUTH_OPEN_THRESHOLD:
                # Mouth center
                mouth_x = int((left_mouth.x + right_mouth.x) / 2 * w)
                mouth_y = int(lower_lip.y * h)

                # Tongue size dynamically based on mouth width
                mouth_width = int(abs(left_mouth.x - right_mouth.x) * w)
                tongue_width = int(mouth_width * 0.8)
                tongue_height = int(tongue_width * 1.2)  # height slightly bigger

                # Overlay tongue
                overlay_png(
                    frame,
                    tongue,
                    mouth_x - tongue_width // 2,
                    mouth_y - int(tongue_height * 0.1),  # slight upward shift
                    (tongue_width, tongue_height)
                )

    cv2.imshow("Dog Filter 🐶😛", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
