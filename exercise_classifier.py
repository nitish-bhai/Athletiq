# exercise_classifier.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ExerciseClassifier:
    def __init__(self, model_path="models/exercise_classifier.pkl"):
        try:
            self.model = joblib.load(model_path)
        except:
            print("⚠️ No pre-trained model found. Train first using trainer.py")
            self.model = None

    def predict(self, landmarks: np.ndarray) -> str:
        """
        Takes flattened pose landmark array (x,y,z for each joint)
        Returns predicted exercise name
        """
        if self.model is None:
            return "Unknown"
        features = landmarks.flatten().reshape(1, -1)
        return self.model.predict(features)[0]
