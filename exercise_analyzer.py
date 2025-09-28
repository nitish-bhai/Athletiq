import cv2
import mediapipe as mp
from utils.angles import calculate_angle
from utils.counters import pushup_counter
from utils.feedback import draw_feedback

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video instead of webcam
cap = cv2.VideoCapture("sample_videos/pushup.mp4")

counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates (shoulder, elbow, wrist)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Count pushups
            counter, stage = pushup_counter(angle, counter, stage)

            # Feedback line (shoulder–wrist)
            is_correct = 70 < angle < 160
            h, w, _ = image.shape
            p1 = (int(shoulder[0] * w), int(shoulder[1] * h))
            p2 = (int(wrist[0] * w), int(wrist[1] * h))
            image = draw_feedback(image, p1, p2, is_correct)

        except:
            pass

        # Render counter
        cv2.putText(image, f'Pushups: {counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Athletiq Exercise Analyzer', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
