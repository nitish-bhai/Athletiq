import cv2
import time

def show_countdown_and_framing(cap, seconds=5):
    """
    Shows countdown overlay + framing guide before capture starts.
    """
    start_time = time.time()
    while time.time() - start_time < seconds:
        ret, frame = cap.read()
        if not ret:
            continue

        # Countdown text
        remaining = seconds - int(time.time() - start_time)
        cv2.putText(frame, f"Starting in {remaining} sec...",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4)

        # Framing guide (center rectangle)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(w*0.25), int(h*0.2)),
                             (int(w*0.75), int(h*0.8)),
                             (0, 255, 0), 3)
        cv2.putText(frame, "Position yourself inside the box",
                    (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Framing Guide", frame)
        cv2.waitKey(100)

    cv2.destroyWindow("Framing Guide")
