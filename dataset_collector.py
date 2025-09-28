# dataset_collector.py
import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

SAVE_DIR = "datasets"

def collect_data(exercise_name: str, num_samples: int = 100):
    cap = cv2.VideoCapture(0)
    os.makedirs(f"{SAVE_DIR}/{exercise_name}", exist_ok=True)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        count = 0
        while cap.isOpened() and count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract landmark array
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)

                # Save sample
                np.save(f"{SAVE_DIR}/{exercise_name}/{exercise_name}_{count}.npy", landmarks)
                count += 1
                cv2.putText(frame, f"Saved sample {count}/{num_samples}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Dataset Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} samples for {exercise_name}")

if __name__ == "__main__":
    exercise = input("Enter exercise name (e.g. pushup, squat, jumping_jack): ")
    collect_data(exercise, num_samples=200)
