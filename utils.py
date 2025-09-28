# utils.py
import numpy as np
import cv2
import mediapipe as mp
import time

LM = mp.solutions.pose.PoseLandmark

# Simple set of connections we draw (pairs of indices)
POSE_CONNECTIONS = [
    (LM.LEFT_SHOULDER.value, LM.LEFT_ELBOW.value),
    (LM.LEFT_ELBOW.value, LM.LEFT_WRIST.value),
    (LM.RIGHT_SHOULDER.value, LM.RIGHT_ELBOW.value),
    (LM.RIGHT_ELBOW.value, LM.RIGHT_WRIST.value),
    (LM.LEFT_HIP.value, LM.LEFT_KNEE.value),
    (LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
    (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value),
    (LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
    (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value),
    (LM.RIGHT_SHOULDER.value, LM.RIGHT_HIP.value),
    (LM.LEFT_SHOULDER.value, LM.RIGHT_SHOULDER.value),
    (LM.LEFT_HIP.value, LM.RIGHT_HIP.value),
]

def calculate_angle(a, b, c):
    """
    a,b,c are (x,y) pixel coordinates
    returns angle at b (in degrees)
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, image_shape):
    """
    Convert MediaPipe landmarks to list of (x,y) pixel coordinates.
    landmarks: results.pose_landmarks.landmark (list)
    image_shape: (h,w,c)
    """
    h, w = image_shape[:2]
    coords = []
    for lm in landmarks:
        coords.append((int(lm.x * w), int(lm.y * h)))
    return coords

def draw_colored_skeleton(frame, coords, correctness_map=None):
    """
    coords: list of (x,y)
    correctness_map: dict landmark_index -> True/False (True means correct)
    draws lines (green if both endpoints correct, red otherwise)
    """
    for (a, b) in POSE_CONNECTIONS:
        if a >= len(coords) or b >= len(coords):
            continue
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        correct = True
        if correctness_map:
            if correctness_map.get(a) is False or correctness_map.get(b) is False:
                correct = False
        color = (0, 255, 0) if correct else (0, 0, 255)
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    # draw keypoints
    for i, (x, y) in enumerate(coords):
        col = (0,255,0)
        if correctness_map and correctness_map.get(i) is False:
            col = (0,0,255)
        cv2.circle(frame, (x,y), 3, col, -1)
    return frame

# Countdown + framing overlay prior to live capture
def show_countdown_and_framing(cap, seconds=5, title_window="Get Ready"):
    """
    Shows a framing rectangle and countdown on the camera feed.
    Blocks until countdown completes or camera fails.
    """
    start_time = time.time()
    while time.time() - start_time < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # countdown
        remaining = seconds - int(time.time() - start_time)
        if remaining < 0:
            remaining = 0
        cv2.putText(frame, f"Starting in {remaining}s", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # framing rectangle (center)
        left = int(w * 0.15)
        right = int(w * 0.85)
        top = int(h * 0.12)
        bottom = int(h * 0.88)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, "Position yourself inside the box", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow(title_window, frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(title_window)

# Utility for inactivity detection (normalized landmarks)
def normalized_landmarks_list(landmarks):
    return [(lm.x, lm.y) for lm in landmarks]

def mean_landmark_displacement(prev, cur):
    if prev is None or cur is None:
        return 1.0
    if len(prev) != len(cur):
        return 1.0
    prev_a = np.array(prev)
    cur_a = np.array(cur)
    d = np.sqrt(np.sum((cur_a - prev_a) ** 2, axis=1))
    return float(np.mean(d))
