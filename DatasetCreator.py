import os
import time
import csv
import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp


DATA_DIR = "dataset"
CLASSES = ["openPalm", "fist", "wShape", "none"]
WINDOW_NAME = "Gesture Dataset Collector"
DRAW = True
MIRROR = True


os.makedirs(DATA_DIR, exist_ok=True)
for c in CLASSES:
    path = os.path.join(DATA_DIR, f"{c}.csv")
    #if class file is not detected, create new
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["sampleId"] + ["label"] + [f"x{i+1}" for i in range(21)] + [f"y{i+1}" for i in range(21)]
            w.writerow(header)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_landmarks(lm_xy, eps=1e-6):
    #gets wrist point
    wrist = lm_xy[0]
    #distance of every point to wrist
    lm_rel = lm_xy - wrist
    #gets the euclidian distance from wrist to middle finger base knuckle
    palm = float(np.linalg.norm(lm_xy[9] - wrist) + eps)
    #normalize by dividing all relative point distances by palm distance
    lm_norm = lm_rel / palm
    return lm_norm, palm


def getLatestSampleId(label):
    path = os.path.join(DATA_DIR, f"{label}.csv")

     # count existing rows (exclude header)
    with open(path, "r") as f:
        existing = sum(1 for ids in f) - 1
    latestSampleId = existing


    return latestSampleId


def write_sample(label, lm_norm):
    path = os.path.join(DATA_DIR, f"{label}.csv")


    sample_id = getLatestSampleId(label)+1


    row = [sample_id] + [label] + lm_norm[:,0].tolist() + lm_norm[:,1].tolist()
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# Picamera2
cam = Picamera2()
video_config = cam.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
cam.configure(video_config)
cam.set_controls({"FrameDurationLimits": (16666, 16666)})
cam.start()
time.sleep(0.2)


curr_label = "openPalm"


print("[i] Controls: 1=open_palm, 2=fist, 3=w_shape, 0=none, r=record, q=quit")


try:
    while True:
        frame = cam.capture_array()
        frame = cv2.flip(frame, 0)


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)


        detected = False
        if res.multi_hand_landmarks:
            detected = True
            img_h, img_w = frame.shape[:2]
            lm = res.multi_hand_landmarks[0]
            pts = np.array([[p.x * img_w, p.y * img_h] for p in lm.landmark], dtype=np.float32)


            norm_xy, scale = normalize_landmarks(pts)


            if DRAW:
                x_min, y_min = pts.min(axis=0).astype(int)
                x_max, y_max = pts.max(axis=0).astype(int)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
                for (px, py) in pts.astype(int):
                    cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)


        cv2.putText(frame, f"Label: {curr_label} Samples: {getLatestSampleId(curr_label)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
        if not detected:
            cv2.putText(frame, "No hand detected", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)


        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(1) & 0xFF


        if k == ord('q'):
            break
        elif k == ord('1'):
            curr_label = "openPalm"
        elif k == ord('2'):
            curr_label = "fist"
        elif k == ord('3'):
            curr_label = "wShape"
        elif k == ord('0'):
            curr_label = "none"
        elif k == ord('r'):
            if res.multi_hand_landmarks:
                write_sample(curr_label, norm_xy)
                print(f"[saved] {curr_label}")
            else:
                print("[warn] cannot save: no hand detected")


finally:
    cam.stop()
    hands.close()
    cv2.destroyAllWindows()



