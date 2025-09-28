import cv2, numpy as np
import mediapipe as mp
from exercise_rules import EXERCISES

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class MultiExerciseAnalyzer:
    def __init__(self):
        self.counters = {ex:0 for ex in EXERCISES.keys()}
        self.stages = {ex:None for ex in EXERCISES.keys()}
        self.feedback = {ex:[] for ex in EXERCISES.keys()}
        self.detected_exercises = set()

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180 else angle

    def analyze(self, frame, landmarks):
        h, w, _ = frame.shape
        points = {lm.name.lower(): [lm.x*w, lm.y*h] for lm in mp_pose.PoseLandmark}

        for ex, rules in EXERCISES.items():
            for joint_set in rules["joints"]:
                a, b, c = [points[j] for j in joint_set]
                angle = self.calculate_angle(a,b,c)

                color = (0,255,0) if rules["good_range"][0] <= angle <= rules["good_range"][1] else (0,0,255)
                cv2.line(frame, tuple(map(int,a)), tuple(map(int,b)), color, 3)
                cv2.line(frame, tuple(map(int,b)), tuple(map(int,c)), color, 3)

                if ex=="pushup":
                    if angle > 160: self.stages[ex] = "up"
                    if angle < 90 and self.stages[ex] == "up":
                        self.stages[ex] = "down"
                        self.counters[ex] += 1
                        self.feedback[ex].append(f"Good rep #{self.counters[ex]}")
                        self.detected_exercises.add(ex)

        return frame
