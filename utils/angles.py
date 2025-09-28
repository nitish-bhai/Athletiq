import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates angle between three points (x,y).
    Angle is measured at point b.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine))

    return angle
