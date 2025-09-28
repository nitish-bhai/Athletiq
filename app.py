# app.py
import streamlit as st
import cv2
import mediapipe as mp
import time
import tempfile
import os
import numpy as np

from utils import calculate_angle, get_landmark_coords, draw_colored_skeleton, show_countdown_and_framing, normalized_landmarks_list
from exercise_rules import evaluate_exercise, EXERCISE_RULES
from report_generator import generate_pdf_report

st.set_page_config(page_title="Athletiq", layout="wide")
st.title("🏋️ Athletiq — Exercise Analyzer")

# Sidebar user info
st.sidebar.header("User information (required for accurate evaluation)")
name = st.sidebar.text_input("Full Name")
age = st.sidebar.number_input("Age", min_value=8, max_value=80, value=18)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, step=1, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, step=1, value=70)
body_type = st.sidebar.selectbox("Body Type", ["slim", "average", "muscular"])

mode = st.sidebar.radio("Mode", ["Auto Detect (recommended)", "Manual select"])
exercise_selected = None
manual_input_mode = None
if mode == "Manual select":
    exercise_selected = st.sidebar.selectbox("Choose exercise to evaluate", sorted(list(EXERCISE_RULES.keys())))
    manual_input_mode = st.sidebar.radio("Input method for this exercise", ["Live Video", "Recorded Video"])

start_button = st.sidebar.button("Start Live Analysis")
stop_button = st.sidebar.button("Stop Analysis")
debug_mode = st.sidebar.checkbox("Show debug results before PDF", value=False)

frame_area = st.empty()
cols = st.columns([2,1])
stats_area = cols[1]

# session flags
if "running" not in st.session_state:
    st.session_state.running = False
if "stop" not in st.session_state:
    st.session_state.stop = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

if start_button:
    st.session_state.running = True
    st.session_state.stop = False
    st.session_state.report_generated = False
if stop_button:
    st.session_state.stop = True
    st.session_state.running = False

# mediapipe config (lenient for uploads)
mp_pose = mp.solutions.pose
POSE_DETECTION_CONF = 0.35
POSE_TRACKING_CONF = 0.35
MODEL_COMPLEXITY = 1

# runtime containers
angle_buffers = {"elbow": [], "knee": [], "hip": [], "arm_up": [], "knee_raise": []}
counters = {ex: 0 for ex in EXERCISE_RULES.keys()}
stages = {ex: None for ex in EXERCISE_RULES.keys()}
detected_exercises = set()
eval_results = {}
metrics = {ex: {"label":"Reps","value":0} for ex in EXERCISE_RULES.keys()}
metrics["plank"]["label"]="Duration (s)"
metrics["rope_jump"]["label"]="Jumps"
exercise_start_time = {}
INACTIVITY_SECONDS = 10.0

# rope detector state (from improved method)
rope_detector = {"alpha":0.25,"smoothed":[],"buf_size":9,"prominence":0.02,"min_frame_gap":6,"last_count_frame":-999}

JOINT_TO_LANDMARKS = {
    "knee":[26,25],"hip":[24,23],"elbow":[14,13],"shoulder":[12,11],"back":[12,24],
    "torso":[11,23],"front_knee":[26],"elbow_up":[14],"arm_up":[16,15],"ankle":[28,27],"knee_raise":[26]
}

def reset_state():
    global angle_buffers, counters, stages, detected_exercises, eval_results, metrics, exercise_start_time, rope_detector
    angle_buffers = {"elbow": [], "knee": [], "hip": [], "arm_up": [], "knee_raise": []}
    for k in counters: counters[k]=0
    for k in stages: stages[k]=None
    detected_exercises=set()
    eval_results={}
    for ex in metrics: metrics[ex]["value"]=0
    exercise_start_time = {}
    rope_detector["smoothed"] = []
    rope_detector["last_count_frame"] = -999

def ema_update(prev, value, alpha):
    if prev is None: return value
    return alpha*value + (1-alpha)*prev

def detect_jump_from_smoothed(smoothed_list, buf_size, prominence, last_count_frame, current_frame_idx, min_frame_gap):
    n = len(smoothed_list)
    if n < buf_size: return False, last_count_frame
    window = smoothed_list[-buf_size:]
    center_idx = buf_size // 2
    center_val = window[center_idx]
    if center_val != min(window): return False, last_count_frame
    left_max = max(window[:center_idx]) if center_idx>0 else center_val
    right_max = max(window[center_idx+1:]) if center_idx+1 < buf_size else center_val
    neighbor_max = max(left_max, right_max)
    prom = neighbor_max - center_val
    if prom >= prominence and (current_frame_idx - last_count_frame) > min_frame_gap:
        last_count_frame = current_frame_idx
        return True, last_count_frame
    return False, last_count_frame

def build_correctness_map_from_eval(eval_res, coords):
    correctness = {}
    if not eval_res or "scores" not in eval_res: return correctness
    for joint, info in eval_res["scores"].items():
        score = info.get("score",0)
        is_correct = score >= 8
        indices = JOINT_TO_LANDMARKS.get(joint, [])
        for idx in indices:
            if 0 <= idx < len(coords):
                correctness[idx] = is_correct
    return correctness

def draw_angle_overlays(frame, coords, measured_angles):
    for joint, angle in measured_angles.items():
        indices = JOINT_TO_LANDMARKS.get(joint, [])
        if not indices: continue
        idx = indices[0]
        if idx < 0 or idx >= len(coords): continue
        x,y = coords[idx]
        cv2.putText(frame, f"{joint}:{int(angle) if angle is not None else '-'}", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# per-frame processing
def process_frame_logic(results, frame, coords_pixel, gender, body_type, prev_norm_lm, frame_idx):
    activity_detected = False
    correctness_map = {}
    h,w = frame.shape[:2]
    coords = coords_pixel
    def safe(i):
        return coords[i] if 0 <= i < len(coords) else (0,0)
    # keypoints
    r_sh = safe(12); r_el = safe(14); r_wr = safe(16)
    l_sh = safe(11); l_el = safe(13); l_wr = safe(15)
    r_hip = safe(24); r_knee = safe(26); r_ank = safe(28)
    l_hip = safe(23); l_knee = safe(25); l_ank = safe(27)
    nose = safe(0)

    # compute angles safely
    elbow_angle = calculate_angle(r_sh, r_el, r_wr) if r_sh and r_el and r_wr else None
    knee_angle = calculate_angle(r_hip, r_knee, r_ank) if r_hip and r_knee and r_ank else None
    hip_angle = calculate_angle(r_sh, r_hip, r_knee) if r_sh and r_hip and r_knee else None
    back_angle = calculate_angle(l_sh, l_hip, l_knee) if l_sh and l_hip and l_knee else None

    # buffers
    if elbow_angle is not None: angle_buffers["elbow"].append(elbow_angle)
    if knee_angle is not None: angle_buffers["knee"].append(knee_angle)
    if hip_angle is not None: angle_buffers["hip"].append(hip_angle)
    angle_buffers["arm_up"].append(abs(r_wr[1] - r_sh[1]))
    angle_buffers["knee_raise"].append(abs(r_knee[1] - r_ank[1]))
    for k in angle_buffers:
        if len(angle_buffers[k])>120: angle_buffers[k].pop(0)

    # decide exercise
    if mode == "Auto Detect (recommended)":
        # simple heuristic
        var = {k:(np.var(v) if len(v)>1 else 0.0) for k,v in angle_buffers.items()}
        if var.get("elbow",0) > 200:
            exercise = "pushup"
        elif var.get("knee",0) > 100:
            exercise = "squat"
        elif var.get("hip",0) > 80 and var.get("knee",0) > 40:
            exercise = "burpee"
        elif var.get("arm_up",0) > 50:
            exercise = "jumping_jack"
        elif var.get("knee_raise",0) > 30:
            exercise = "high_knees"
        else:
            exercise = list(EXERCISE_RULES.keys())[0]
    else:
        exercise = exercise_selected

    measured = {}
    eval_res = None

    # rope-jump (improved)
    if exercise == "rope_jump":
        # normalized ankles via MediaPipe landmarks
        # we access normalized list via results.pose_landmarks.landmark
        l_ank_y = results.pose_landmarks.landmark[27].y if len(results.pose_landmarks.landmark)>27 else None
        r_ank_y = results.pose_landmarks.landmark[28].y if len(results.pose_landmarks.landmark)>28 else None
        counted = False
        if l_ank_y is not None and r_ank_y is not None:
            sample = (l_ank_y + r_ank_y)/2.0
            prev = rope_detector["smoothed"][-1] if rope_detector["smoothed"] else None
            s = ema_update(prev, sample, rope_detector["alpha"])
            rope_detector["smoothed"].append(s)
            if len(rope_detector["smoothed"]) > 300:
                rope_detector["smoothed"] = rope_detector["smoothed"][-300:]
            counted, rope_detector["last_count_frame"] = detect_jump_from_smoothed(
                rope_detector["smoothed"],
                rope_detector["buf_size"],
                rope_detector["prominence"],
                rope_detector["last_count_frame"],
                frame_idx,
                rope_detector["min_frame_gap"]
            )
            if counted:
                counters["rope_jump"] += 1
                metrics["rope_jump"]["value"] = counters["rope_jump"]
        else:
            # fallback knee method
            if knee_angle is not None:
                if knee_angle < 150:
                    if stages.get("rope_jump") != "jump":
                        stages["rope_jump"] = "jump"
                else:
                    if stages.get("rope_jump") == "jump":
                        counters["rope_jump"] += 1
                        metrics["rope_jump"]["value"] = counters["rope_jump"]
        measured = {"knee": knee_angle} if knee_angle is not None else {}
        eval_res = evaluate_exercise("rope_jump", gender, body_type, measured)
        detected_exercises.add("rope_jump")
        eval_results["rope_jump"] = {"reps": counters["rope_jump"], "eval": eval_res}
        activity_detected = True

    # squat
    elif exercise == "squat":
        measured = {"knee": knee_angle, "hip": hip_angle, "back": back_angle}
        eval_res = evaluate_exercise("squat", gender, body_type, measured)
        if knee_angle is not None and knee_angle < 110:
            stages["squat"] = "down"
        if knee_angle is not None and stages.get("squat") == "down" and knee_angle > 155:
            counters["squat"] += 1
            metrics["squat"]["value"] = counters["squat"]
        detected_exercises.add("squat")
        eval_results["squat"] = {"reps": counters["squat"], "eval": eval_res}
        activity_detected = True

    # pushup
    elif exercise == "pushup":
        shoulder_angle = calculate_angle(r_el, r_sh, r_hip) if r_el and r_sh and r_hip else None
        measured = {"elbow": elbow_angle, "shoulder": shoulder_angle, "back": back_angle}
        eval_res = evaluate_exercise("pushup", gender, body_type, measured)
        if elbow_angle is not None and elbow_angle < 95:
            stages["pushup"] = "down"
        if elbow_angle is not None and stages.get("pushup") == "down" and elbow_angle > 150:
            counters["pushup"] += 1
            metrics["pushup"]["value"] = counters["pushup"]
        detected_exercises.add("pushup")
        eval_results["pushup"] = {"reps": counters["pushup"], "eval": eval_res}
        activity_detected = True

    # plank (duration)
    elif exercise == "plank":
        measured = {"spine": back_angle, "hip": hip_angle}
        eval_res = evaluate_exercise("plank", gender, body_type, measured)
        if "plank" not in exercise_start_time:
            exercise_start_time["plank"] = time.time()
        metrics["plank"]["value"] = int(time.time() - exercise_start_time["plank"])
        detected_exercises.add("plank")
        eval_results["plank"] = {"reps": 0, "eval": eval_res}
        activity_detected = True

    # jumping_jack
    elif exercise == "jumping_jack":
        left_up = abs(l_wr[1] - l_sh[1])
        right_up = abs(r_wr[1] - r_sh[1])
        arm_up = (left_up < 0.25*frame.shape[0]) or (right_up < 0.25*frame.shape[0])
        if arm_up:
            if stages.get("jumping_jack") != "up":
                stages["jumping_jack"] = "up"
        else:
            if stages.get("jumping_jack") == "up":
                counters["jumping_jack"] += 1
                metrics["jumping_jack"]["value"] = counters["jumping_jack"]
        measured = {"arm_up": 150 if arm_up else 10, "leg_out": abs(l_ank[0] - r_ank[0]) if len(coords)>27 else 0}
        eval_res = evaluate_exercise("jumping_jack", gender, body_type, measured)
        detected_exercises.add("jumping_jack")
        eval_results["jumping_jack"] = {"reps": counters["jumping_jack"], "eval": eval_res}
        activity_detected = True

    # situp
    elif exercise == "situp":
        torso_angle = calculate_angle(l_hip, l_sh, l_knee) if l_hip and l_sh and l_knee else None
        measured = {"torso": torso_angle}
        if torso_angle is not None and torso_angle < 80:
            if stages.get("situp") != "up":
                stages["situp"] = "up"
        else:
            if stages.get("situp") == "up":
                counters["situp"] += 1
                metrics["situp"]["value"] = counters["situp"]
        eval_res = evaluate_exercise("situp", gender, body_type, measured)
        detected_exercises.add("situp")
        eval_results["situp"] = {"reps": counters["situp"], "eval": eval_res}
        activity_detected = True

    # pullup
    elif exercise == "pullup":
        if nose[1] < r_wr[1] - 20:
            if stages.get("pullup") != "up":
                stages["pullup"] = "up"
        else:
            if stages.get("pullup") == "up":
                counters["pullup"] += 1
                metrics["pullup"]["value"] = counters["pullup"]
        measured = {"elbow_up": elbow_angle}
        eval_res = evaluate_exercise("pullup", gender, body_type, measured)
        detected_exercises.add("pullup")
        eval_results["pullup"] = {"reps": counters["pullup"], "eval": eval_res}
        activity_detected = True

    # lunge
    elif exercise == "lunge":
        front_knee_angle = calculate_angle(r_hip, r_knee, r_ank) if r_hip and r_knee and r_ank else None
        if front_knee_angle is not None and front_knee_angle < 100:
            if stages.get("lunge") != "down":
                stages["lunge"] = "down"
        else:
            if stages.get("lunge") == "down":
                counters["lunge"] += 1
                metrics["lunge"]["value"] = counters["lunge"]
        measured = {"front_knee": front_knee_angle}
        eval_res = evaluate_exercise("lunge", gender, body_type, measured)
        detected_exercises.add("lunge")
        eval_results["lunge"] = {"reps": counters["lunge"], "eval": eval_res}
        activity_detected = True

    # burpee
    elif exercise == "burpee":
        if hip_angle is not None and hip_angle < 90:
            stages["burpee"] = "down"
        if hip_angle is not None and stages.get("burpee") == "down" and hip_angle > 160:
            counters["burpee"] += 1
            metrics["burpee"]["value"] = counters["burpee"]
        measured = {"hip": hip_angle}
        eval_res = evaluate_exercise("burpee", gender, body_type, measured)
        detected_exercises.add("burpee")
        eval_results["burpee"] = {"reps": counters["burpee"], "eval": eval_res}
        activity_detected = True

    # deadlift
    elif exercise == "deadlift":
        measured = {"hip": hip_angle, "back": back_angle}
        if hip_angle is not None and hip_angle < 100:
            stages["deadlift"] = "down"
        if hip_angle is not None and stages.get("deadlift") == "down" and hip_angle > 150:
            counters["deadlift"] += 1
            metrics["deadlift"]["value"] = counters["deadlift"]
        eval_res = evaluate_exercise("deadlift", gender, body_type, measured)
        detected_exercises.add("deadlift")
        eval_results["deadlift"] = {"reps": counters["deadlift"], "eval": eval_res}
        activity_detected = True

    # shoulder_press
    elif exercise == "shoulder_press":
        elbow = calculate_angle(r_sh, r_el, r_wr) if r_sh and r_el and r_wr else None
        if elbow is not None and elbow > 150:
            if stages.get("shoulder_press") != "up":
                counters["shoulder_press"] += 1
                metrics["shoulder_press"]["value"] = counters["shoulder_press"]
        measured = {"elbow": elbow}
        eval_res = evaluate_exercise("shoulder_press", gender, body_type, measured)
        detected_exercises.add("shoulder_press")
        eval_results["shoulder_press"] = {"reps": counters["shoulder_press"], "eval": eval_res}
        activity_detected = True

    # bicep_curl
    elif exercise == "bicep_curl":
        elbow = calculate_angle(r_sh, r_el, r_wr) if r_sh and r_el and r_wr else None
        if elbow is not None and elbow < 50:
            if stages.get("bicep_curl") != "up":
                counters["bicep_curl"] += 1
                metrics["bicep_curl"]["value"] = counters["bicep_curl"]
        measured = {"elbow": elbow}
        eval_res = evaluate_exercise("bicep_curl", gender, body_type, measured)
        detected_exercises.add("bicep_curl")
        eval_results["bicep_curl"] = {"reps": counters["bicep_curl"], "eval": eval_res}
        activity_detected = True

    # tricep_dip
    elif exercise == "tricep_dip":
        elbow = calculate_angle(r_sh, r_el, r_wr) if r_sh and r_el and r_wr else None
        if elbow is not None and elbow < 90:
            if stages.get("tricep_dip") != "down":
                stages["tricep_dip"] = "down"
        if elbow is not None and stages.get("tricep_dip") == "down" and elbow > 140:
            counters["tricep_dip"] += 1
            metrics["tricep_dip"]["value"] = counters["tricep_dip"]
        measured = {"elbow": elbow}
        eval_res = evaluate_exercise("tricep_dip", gender, body_type, measured)
        detected_exercises.add("tricep_dip")
        eval_results["tricep_dip"] = {"reps": counters["tricep_dip"], "eval": eval_res}
        activity_detected = True

    # mountain_climber
    elif exercise == "mountain_climber":
        knee_val = knee_angle
        if knee_val is not None and knee_val < 70:
            stages["mountain_climber"] = "in"
        if knee_val is not None and stages.get("mountain_climber") == "in" and knee_val > 120:
            counters["mountain_climber"] += 1
            metrics["mountain_climber"]["value"]=counters["mountain_climber"]
        measured = {"knee": knee_val}
        eval_res = evaluate_exercise("mountain_climber", gender, body_type, measured)
        detected_exercises.add("mountain_climber")
        eval_results["mountain_climber"] = {"reps": counters["mountain_climber"], "eval": eval_res}
        activity_detected = True

    # high_knees
    elif exercise == "high_knees":
        knee_val = knee_angle
        if knee_val is not None and knee_val < 70:
            if stages.get("high_knees") != "up":
                counters["high_knees"] += 1
                metrics["high_knees"]["value"] = counters["high_knees"]
        measured = {"knee_raise": knee_val}
        eval_res = evaluate_exercise("high_knees", gender, body_type, measured)
        detected_exercises.add("high_knees")
        eval_results["high_knees"] = {"reps": counters["high_knees"], "eval": eval_res}
        activity_detected = True

    # draw overlays: angle text + colored skeleton from correctness
    if eval_res:
        correctness_map = build_correctness_map_from_eval(eval_res, coords)
    if measured:
        draw_angle_overlays(frame, coords, measured)

    frame = draw_colored_skeleton(frame, coords, correctness_map if correctness_map else None)
    cur_label = metrics.get(exercise, {}).get("label", "Reps")
    cur_value = metrics.get(exercise, {}).get("value", 0)
    try:
        cv2.putText(frame, f"{cur_label}: {cur_value}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Exercise: {exercise}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Reps: {counters.get(exercise,0)}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    except Exception:
        pass

    return frame, normalized_landmarks_list(results.pose_landmarks.landmark) if results.pose_landmarks else None, activity_detected

# UPLOAD branch
uploaded_file = None
if mode == "Manual select" and manual_input_mode == "Recorded Video":
    st.info(f"Manual → Recorded Video selected for '{exercise_selected}'. Use the uploader below.")
    uploaded_file = st.file_uploader("Upload recorded video (mp4 / mov / avi)", type=["mp4","mov","avi"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Analyze Uploaded Video"):
            reset_state()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                tfile.write(uploaded_file.read())
                tfile.flush(); tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                prev_norm = None
                last_active_time = time.time()
                frame_idx = 0
                with mp_pose.Pose(min_detection_confidence=POSE_DETECTION_CONF, min_tracking_confidence=POSE_TRACKING_CONF, model_complexity=MODEL_COMPLEXITY) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_idx += 1
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)
                        if results.pose_landmarks:
                            coords = get_landmark_coords(results.pose_landmarks.landmark, frame.shape)
                            frame, prev_norm, active = process_frame_logic(results, frame, coords, gender, body_type, prev_norm, frame_idx)
                            if active:
                                last_active_time = time.time()
                        frame_area.image(frame, channels="BGR")
                        if st.session_state.stop:
                            break
                cap.release()
                # remove temp safely
                for _ in range(8):
                    try:
                        os.remove(tfile.name); break
                    except PermissionError:
                        time.sleep(0.12)
                st.success("Uploaded video analysis finished.")
            except Exception as e:
                try: cap.release()
                except: pass
                try: tfile.close()
                except: pass
                try: os.remove(tfile.name)
                except: pass
                st.error(f"Error: {e}")

# LIVE camera branch
if st.session_state.running:
    reset_state()
    st.info("Starting live camera — press Stop Analysis in the sidebar to stop")
    cap = cv2.VideoCapture(0)
    show_countdown_and_framing(cap, seconds=5, title_window="Get Ready")
    prev_norm = None
    last_active_time = time.time()
    frame_idx = 0
    with mp_pose.Pose(min_detection_confidence=POSE_DETECTION_CONF, min_tracking_confidence=POSE_TRACKING_CONF, model_complexity=MODEL_COMPLEXITY) as pose:
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                coords = get_landmark_coords(results.pose_landmarks.landmark, frame.shape)
                frame, prev_norm, active = process_frame_logic(results, frame, coords, gender, body_type, prev_norm, frame_idx)
                if active:
                    last_active_time = time.time()
            frame_area.image(frame, channels="BGR")
            # live stats
            with stats_area:
                st.markdown("## Live Stats")
                for ex in sorted(metrics.keys()):
                    val = metrics[ex]["value"]
                    if val:
                        st.write(f"**{ex}** — {metrics[ex]['label']}: {val}")
                st.write("Detected: " + ", ".join(sorted(detected_exercises)))
                idle_sec = time.time() - last_active_time
                st.write(f"Idle (s): {idle_sec:.1f}")
            if time.time() - last_active_time > INACTIVITY_SECONDS:
                st.info("No activity detected for 10s — stopping and generating report.")
                st.session_state.stop = True
                st.session_state.running = False
                break
            time.sleep(0.02)
    cap.release()
    cv2.destroyAllWindows()

# When stopped -> build final results and generate PDF
if st.session_state.stop and not st.session_state.report_generated:
    user_info = {"name": name or "Unknown", "age": age, "gender": gender, "body_type": body_type, "height": height, "weight": weight}
    results = {}
    for ex in sorted(set(list(metrics.keys()))):
        metric_label = metrics[ex]["label"]
        metric_value = metrics[ex]["value"]
        entry = {"metric_label": metric_label, "metric_value": metric_value, "reps": counters.get(ex,0)}
        if ex in eval_results:
            entry["eval"] = eval_results[ex]["eval"]
        results[ex] = entry

    # keep only exercises with some activity / metrics
    filtered = {k:v for k,v in results.items() if (v.get("metric_value",0) or k in detected_exercises)}
    if not filtered:
        filtered = {"no_activity":{"metric_label":"N/A","metric_value":0,"reps":0,"eval":{"scores":{},"average_score":0}}}

    if debug_mode:
        st.write("DEBUG: final results sent to PDF generator:")
        st.write(filtered)

    pdf_path = generate_pdf_report(user_info, filtered)
    st.success("Final report generated")
    with open(pdf_path, "rb") as f:
        st.download_button("Download final report", f, file_name=os.path.basename(pdf_path))
    st.session_state.report_generated = True
