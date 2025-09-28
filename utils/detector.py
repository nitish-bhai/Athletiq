def detect_exercise(landmarks):
    """
    Simple heuristic: detect which exercise is being performed.
    Returns one of: pushup, squat, jump, pullup, run
    """

    # Extract key coordinates
    left_shoulder = landmarks[11].y
    left_hip = landmarks[23].y
    left_knee = landmarks[25].y
    left_ankle = landmarks[27].y

    torso_length = abs(left_shoulder - left_hip)
    leg_length = abs(left_hip - left_knee)

    # Heuristic rules
    if torso_length < 0.25 and leg_length > 0.35:
        return "pushup"
    elif leg_length < 0.25:
        return "squat"
    elif abs(left_hip - left_ankle) > 0.5:
        return "jump"
    elif torso_length > 0.4 and leg_length > 0.4:
        return "pullup"
    else:
        return "run"
