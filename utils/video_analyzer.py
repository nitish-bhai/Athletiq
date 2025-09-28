import cv2
import time

def detect_idle(cap, idle_threshold=10):
    """
    Detects if no activity for X seconds (based on frame differences).
    Returns True if idle detected.
    """
    prev_frame = None
    idle_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = cv2.countNonZero(thresh)

        if motion_score < 500:  # No motion
            if idle_start is None:
                idle_start = time.time()
            elif time.time() - idle_start >= idle_threshold:
                return True
        else:
            idle_start = None

        prev_frame = gray

        cv2.imshow("Live Analysis", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    return False
