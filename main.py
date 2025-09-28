import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter = 0
stage = None

# Open webcam (0) or change path to a video file
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Example: squat counter using hip & knee
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

            if hip.y > knee.y:
                stage = "down"
            if hip.y < knee.y and stage == "down":
                stage = "up"
                counter += 1
                print(f"Reps: {counter}")

        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        )

        # Show counter on screen
        cv2.putText(image, f"Reps: {counter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Pose Detection", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Save feedback
with open("feedback.txt", "w", encoding="utf-8") as f:
    f.write("Form Quality: ✅ Good\n" if counter > 5 else "Form Quality: ❌ Needs Improvement\n")
