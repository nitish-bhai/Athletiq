# trainer.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_DIR = "datasets"   # each exercise folder should contain .npy landmark arrays
MODEL_PATH = "models/exercise_classifier.pkl"

def load_dataset():
    X, y = [], []
    for label in os.listdir(DATASET_DIR):
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                data = np.load(os.path.join(label_dir, file))
                X.append(data.flatten())
                y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("📥 Loading dataset...")
    X, y = load_dataset()

    print("🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Training model...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    print("✅ Training complete. Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    print("📊 Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
