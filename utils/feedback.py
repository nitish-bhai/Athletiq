import cv2

def draw_feedback(frame, point1, point2, is_correct):
    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green/Red
    cv2.line(frame, point1, point2, color, 4)
    return frame
