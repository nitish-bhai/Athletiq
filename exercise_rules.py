# exercise_rules.py
"""
Rule-based thresholds and scoring for 15 exercises.
Provides evaluate_exercise(exercise, gender, body_type, measured_angles)
measured_angles: dict mapping joint names used by each exercise to measured angle or metric
"""

EXERCISE_RULES = {
    "squat": {
        "male": {
            "slim": {"knee": (80, 100), "hip": (70, 95), "back": (165, 180)},
            "average": {"knee": (85, 105), "hip": (75, 100), "back": (160, 180)},
            "muscular": {"knee": (90, 110), "hip": (80, 105), "back": (155, 180)},
        },
        "female": {
            "slim": {"knee": (75, 95), "hip": (70, 90), "back": (165, 180)},
            "average": {"knee": (80, 100), "hip": (75, 95), "back": (160, 180)},
            "muscular": {"knee": (85, 105), "hip": (80, 100), "back": (155, 180)},
        },
    },

    "pushup": {
        "male": {
            "slim": {"elbow": (80, 100), "shoulder": (50, 70), "back": (165, 180)},
            "average": {"elbow": (85, 105), "shoulder": (55, 75), "back": (160, 180)},
            "muscular": {"elbow": (90, 110), "shoulder": (60, 80), "back": (155, 180)},
        },
        "female": {
            "slim": {"elbow": (75, 95), "shoulder": (50, 70), "back": (165, 180)},
            "average": {"elbow": (80, 100), "shoulder": (55, 75), "back": (160, 180)},
            "muscular": {"elbow": (85, 105), "shoulder": (60, 80), "back": (155, 180)},
        },
    },

    "plank": {
        "male": {
            "slim": {"spine": (165, 180), "hip": (160, 180)},
            "average": {"spine": (160, 180), "hip": (155, 180)},
            "muscular": {"spine": (155, 180), "hip": (150, 180)},
        },
        "female": {
            "slim": {"spine": (165, 180), "hip": (160, 180)},
            "average": {"spine": (160, 180), "hip": (155, 180)},
            "muscular": {"spine": (155, 180), "hip": (150, 180)},
        },
    },

    "jumping_jack": {
        "male": {
            "slim": {"arm_up": (150, 180), "leg_out": (20, 60)},
            "average": {"arm_up": (140, 175), "leg_out": (20, 55)},
            "muscular": {"arm_up": (130, 170), "leg_out": (15, 50)},
        },
        "female": {
            "slim": {"arm_up": (150, 180), "leg_out": (20, 60)},
            "average": {"arm_up": (140, 175), "leg_out": (20, 55)},
            "muscular": {"arm_up": (130, 170), "leg_out": (15, 50)},
        },
    },

    "situp": {
        "male": {
            "slim": {"torso": (50, 90)},
            "average": {"torso": (45, 85)},
            "muscular": {"torso": (40, 80)},
        },
        "female": {
            "slim": {"torso": (50, 90)},
            "average": {"torso": (45, 85)},
            "muscular": {"torso": (40, 80)},
        },
    },

    "pullup": {
        "male": {
            "slim": {"elbow_up": (50, 80), "elbow_down": (150, 180)},
            "average": {"elbow_up": (55, 85), "elbow_down": (145, 180)},
            "muscular": {"elbow_up": (60, 90), "elbow_down": (140, 175)},
        },
        "female": {
            "slim": {"elbow_up": (50, 80), "elbow_down": (150, 180)},
            "average": {"elbow_up": (55, 85), "elbow_down": (145, 180)},
            "muscular": {"elbow_up": (60, 90), "elbow_down": (140, 175)},
        },
    },

    "rope_jump": {
        "male": {
            "slim": {"knee": (140, 180), "ankle": (140, 180)},
            "average": {"knee": (140, 180), "ankle": (140, 180)},
            "muscular": {"knee": (140, 180), "ankle": (140, 180)},
        },
        "female": {
            "slim": {"knee": (140, 180), "ankle": (140, 180)},
            "average": {"knee": (140, 180), "ankle": (140, 180)},
            "muscular": {"knee": (140, 180), "ankle": (140, 180)},
        },
    },

    "lunge": {
        "male": {
            "slim": {"front_knee": (80, 100), "back_knee": (140, 180)},
            "average": {"front_knee": (85, 105), "back_knee": (140, 180)},
            "muscular": {"front_knee": (90, 110), "back_knee": (135, 180)},
        },
        "female": {
            "slim": {"front_knee": (75, 95), "back_knee": (140, 180)},
            "average": {"front_knee": (80, 100), "back_knee": (140, 180)},
            "muscular": {"front_knee": (85, 105), "back_knee": (135, 180)},
        },
    },

    "burpee": {
        "male": {
            "slim": {"hip": (70, 100), "knee": (80, 100)},
            "average": {"hip": (75, 105), "knee": (85, 105)},
            "muscular": {"hip": (80, 110), "knee": (90, 110)},
        },
        "female": {
            "slim": {"hip": (70, 95), "knee": (75, 95)},
            "average": {"hip": (75, 100), "knee": (80, 100)},
            "muscular": {"hip": (80, 105), "knee": (85, 105)},
        },
    },

    "deadlift": {
        "male": {
            "slim": {"hip": (70, 95), "back": (160, 180)},
            "average": {"hip": (75, 100), "back": (155, 180)},
            "muscular": {"hip": (80, 105), "back": (150, 180)},
        },
        "female": {
            "slim": {"hip": (70, 90), "back": (160, 180)},
            "average": {"hip": (75, 95), "back": (155, 180)},
            "muscular": {"hip": (80, 100), "back": (150, 180)},
        },
    },

    "shoulder_press": {
        "male": {
            "slim": {"elbow": (150, 180), "shoulder": (80, 110)},
            "average": {"elbow": (145, 180), "shoulder": (85, 115)},
            "muscular": {"elbow": (140, 180), "shoulder": (90, 120)},
        },
        "female": {
            "slim": {"elbow": (150, 180), "shoulder": (75, 105)},
            "average": {"elbow": (145, 180), "shoulder": (80, 110)},
            "muscular": {"elbow": (140, 180), "shoulder": (85, 115)},
        },
    },

    "bicep_curl": {
        "male": {
            "slim": {"elbow": (30, 70)},
            "average": {"elbow": (35, 75)},
            "muscular": {"elbow": (40, 80)},
        },
        "female": {
            "slim": {"elbow": (30, 65)},
            "average": {"elbow": (35, 70)},
            "muscular": {"elbow": (40, 75)},
        },
    },

    "tricep_dip": {
        "male": {
            "slim": {"elbow": (80, 100), "shoulder": (40, 70)},
            "average": {"elbow": (85, 105), "shoulder": (45, 75)},
            "muscular": {"elbow": (90, 110), "shoulder": (50, 80)},
        },
        "female": {
            "slim": {"elbow": (75, 95), "shoulder": (40, 65)},
            "average": {"elbow": (80, 100), "shoulder": (45, 70)},
            "muscular": {"elbow": (85, 105), "shoulder": (50, 75)},
        },
    },

    "mountain_climber": {
        "male": {
            "slim": {"knee": (60, 90), "hip": (70, 100)},
            "average": {"knee": (65, 95), "hip": (75, 105)},
            "muscular": {"knee": (70, 100), "hip": (80, 110)},
        },
        "female": {
            "slim": {"knee": (55, 85), "hip": (70, 95)},
            "average": {"knee": (60, 90), "hip": (75, 100)},
            "muscular": {"knee": (65, 95), "hip": (80, 105)},
        },
    },

    "high_knees": {
        "male": {
            "slim": {"knee_raise": (60, 110)},
            "average": {"knee_raise": (65, 115)},
            "muscular": {"knee_raise": (70, 120)},
        },
        "female": {
            "slim": {"knee_raise": (60, 110)},
            "average": {"knee_raise": (65, 115)},
            "muscular": {"knee_raise": (70, 120)},
        },
    },
}

# --- scoring helpers ---
def score_angle(angle, expected_range):
    """Return score (0–10) based on closeness to expected range."""
    low, high = expected_range
    if angle is None:
        return 0
    if low <= angle <= high:
        return 10
    # distance from nearest bound (smaller is better)
    if angle < low:
        dist = low - angle
    else:
        dist = angle - high
    if dist <= 5:
        return 8
    if dist <= 10:
        return 6
    if dist <= 20:
        return 3
    return 1

def evaluate_exercise(exercise, gender, body_type, measured_angles):
    """
    measured_angles: dict of joint_name -> numeric angle/metric
    returns {"scores": {joint: {...}}, "average_score": float}
    """
    out = {"scores": {}, "average_score": 0.0}
    if exercise not in EXERCISE_RULES:
        return out
    # fallback handling if provided profile not found
    try:
        rules = EXERCISE_RULES[exercise][gender][body_type]
    except KeyError:
        # pick first available gender/profile
        g = list(EXERCISE_RULES[exercise].keys())[0]
        p = list(EXERCISE_RULES[exercise][g].keys())[0]
        rules = EXERCISE_RULES[exercise][g][p]

    total = 0
    count = 0
    for joint, rng in rules.items():
        val = measured_angles.get(joint)
        s = score_angle(val, rng) if val is not None else 0
        out["scores"][joint] = {"angle": int(val) if val is not None else None, "expected": rng, "score": s}
        total += s
        count += 1
    out["average_score"] = round(total / count, 2) if count > 0 else 0.0
    return out
