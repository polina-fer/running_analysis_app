import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pandas as pd
import joblib
import os

X_features = [
    'cadence_value (spm)', 'gct_value (ms)', 'vert_osc_value (%)',
    'knee_strike_angle (deg)', 'knee_push_angle (deg)', 'elbow_angle_val (deg)',
    'leg_split_val (deg)', 'trunk_lean_value (deg)',
    'n_ankle_x', 'n_ankle_y', 'n_knee_x', 'n_knee_y',
    'n_foot_stretch', 'n_heel_toe_slope', 'n_knee_elevation',
    'n_shoulder_lean', 'n_elbow_x'
]

mp_pose = mp.solutions.pose
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'side_view_models.joblib')
models = joblib.load(MODEL_PATH)


def calculate_angle(point_a, point_b, point_c):
    point_a, point_b, point_c = np.array(point_a), np.array(point_b), np.array(point_c)
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    cosine_angle = np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def prepare_ml_features(lm, side, frame_width, frame_height):
    if side == "Right":
        hip, knee, ankle = 24, 26, 28
        shoulder, elbow, wrist = 12, 14, 16
        heel, toe = 30, 32
    else:
        hip, knee, ankle = 23, 25, 27
        shoulder, elbow, wrist = 11, 13, 15
        heel, toe = 29, 31

    mid_hip_x = (lm[24].x + lm[23].x) / 2
    mid_hip_y = (lm[24].y + lm[23].y) / 2
    mid_shoulder_x = (lm[12].x + lm[11].x) / 2
    mid_shoulder_y = (lm[12].y + lm[11].y) / 2

    torso_dist = np.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
    if torso_dist == 0:
        torso_dist = 1

    def norm_x(l_idx): return (lm[l_idx].x - mid_hip_x) / torso_dist
    def norm_y(l_idx): return (lm[l_idx].y - mid_hip_y) / torso_dist

    return {
        'n_ankle_x': round(norm_x(ankle), 4),
        'n_ankle_y': round(norm_y(ankle), 4),
        'n_knee_x': round(norm_x(knee), 4),
        'n_knee_y': round(norm_y(knee), 4),
        'n_foot_stretch': round(norm_x(ankle) - norm_x(hip), 4),
        'n_heel_toe_slope': round(lm[heel].y - lm[toe].y, 4),
        'n_knee_elevation': round(norm_y(knee), 4),
        'n_shoulder_lean': round(norm_x(shoulder), 4),
        'n_elbow_x': round(norm_x(elbow), 4)
    }


def analyze_video(video_path, output_path=None):
    video_capture = cv2.VideoCapture(video_path)
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    orig_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_width > 640:
        scale = 640 / orig_width
    else:
        scale = 1.0

    frame_width = int(orig_width * scale)
    frame_height = int(orig_height * scale)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, frames_per_second, (frame_width, frame_height))

    push_off_threshold = 0.08
    min_swing_frames = frames_per_second * 0.22
    min_contact_frames = 3

    all_step_metrics_storage = []
    history_window_size = 10
    right_knee_angle_history = deque(maxlen=history_window_size)
    left_knee_angle_history = deque(maxlen=history_window_size)
    hip_height_history = deque(maxlen=50)
    cadence_history = deque(maxlen=5)
    gct_filter_history = deque(maxlen=5)

    current_max_split = 0
    leg_is_on_ground = {"Right": False, "Left": False}
    ankle_y_at_contact = {"Right": None, "Left": None}
    last_ankle_y = {"Right": None, "Left": None}
    last_leg_that_landed = None
    last_strike_frame_index = 0
    previous_strike_frame = None
    avg_cadence = 0
    right_step_count = 0
    left_step_count = 0
    current_status_event = ""
    latest_ai_results = {}
    last_display_frame = None
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=0) as pose_analyzer:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1

            if scale != 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height))

            if frame_count % 2 != 0:
                if writer and last_display_frame is not None:
                    writer.write(last_display_frame)
                continue

            overlay_layer = frame.copy()
            display_frame = frame.copy()
            fh, fw, _ = frame.shape
            results = pose_analyzer.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_pixel_point(idx):
                    return np.array([int(landmarks[idx].x * fw), int(landmarks[idx].y * fh)])

                r_hip, r_knee, r_ankle = get_pixel_point(24), get_pixel_point(26), get_pixel_point(28)
                l_hip, l_knee, l_ankle = get_pixel_point(23), get_pixel_point(25), get_pixel_point(27)
                r_shld, r_elb, r_wrst = get_pixel_point(12), get_pixel_point(14), get_pixel_point(16)
                l_shld, l_elb, l_wrst = get_pixel_point(11), get_pixel_point(13), get_pixel_point(15)

                r_knee_ang = calculate_angle(r_hip, r_knee, r_ankle)
                l_knee_ang = calculate_angle(l_hip, l_knee, l_ankle)
                r_elb_ang = calculate_angle(r_shld, r_elb, r_wrst)
                l_elb_ang = calculate_angle(l_shld, l_elb, l_wrst)

                trunk_vector = np.array([landmarks[12].x - landmarks[24].x, landmarks[12].y - landmarks[24].y])
                trunk_angle = 180 - np.degrees(np.arctan2(np.abs(trunk_vector[0]), np.abs(trunk_vector[1])))

                torso_h = np.abs(landmarks[24].y - landmarks[12].y)
                mid_h_y = (landmarks[24].y + landmarks[23].y) / 2
                hip_height_history.append(mid_h_y)
                v_osc = ((max(hip_height_history) - min(hip_height_history)) / torso_h) * 100 if torso_h > 0 else 0

                v_r, v_l = (r_knee - r_hip), (l_knee - l_hip)
                split_angle = np.degrees(np.arccos(np.clip(
                    np.dot(v_r / np.linalg.norm(v_r), v_l / np.linalg.norm(v_l)), -1.0, 1.0)))
                if split_angle > current_max_split:
                    current_max_split = split_angle

                # draw overlays
                torso_pts = np.array([r_shld, l_shld, l_hip, r_hip], np.int32)
                cv2.fillPoly(overlay_layer, [torso_pts], (0, 255, 0))
                cv2.addWeighted(overlay_layer, 0.15, display_frame, 0.85, 0, display_frame)
                cv2.polylines(display_frame, [torso_pts], True, (255, 255, 255), 2)
                cv2.line(display_frame, tuple(r_shld), tuple(l_hip), (255, 255, 255), 1)
                cv2.line(display_frame, tuple(l_shld), tuple(r_hip), (255, 255, 255), 1)
                cv2.line(display_frame, tuple(r_ankle), tuple(l_ankle), (255, 0, 255), 2)

                for h, k, a, c in [(r_hip, r_knee, r_ankle, (0, 255, 0)), (l_hip, l_knee, l_ankle, (0, 255, 255))]:
                    cv2.line(display_frame, tuple(h), tuple(k), c, 3)
                    cv2.line(display_frame, tuple(k), tuple(a), c, 3)
                for s, e, w in [(r_shld, r_elb, r_wrst), (l_shld, l_elb, l_wrst)]:
                    cv2.line(display_frame, tuple(s), tuple(e), (255, 165, 0), 3)
                    cv2.line(display_frame, tuple(e), tuple(w), (255, 165, 0), 3)

                for j, ang, c in [(r_knee, r_knee_ang, (0, 255, 0)), (l_knee, l_knee_ang, (0, 255, 255)), (r_elb, r_elb_ang, (255, 165, 0))]:
                    cv2.putText(display_frame, f"{int(ang)}", tuple(j + [10, -10]), 1, 1.2, c, 2)

                right_knee_angle_history.append(r_knee_ang)
                left_knee_angle_history.append(l_knee_ang)
                current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

                if len(right_knee_angle_history) == history_window_size:
                    mid_idx = history_window_size // 2
                    lockout = frames_per_second * 0.18 if avg_cadence > 190 else min_swing_frames

                    for side, history, ankle_y, e_ang in [
                        ("Right", right_knee_angle_history, landmarks[28].y, r_elb_ang),
                        ("Left", left_knee_angle_history, landmarks[27].y, l_elb_ang)
                    ]:
                        if (last_leg_that_landed != side and
                                history[mid_idx] > 150 and
                                history[mid_idx] == max(history) and
                                (current_frame - last_strike_frame_index) > lockout):

                            other_side = "Left" if side == "Right" else "Right"
                            if leg_is_on_ground[other_side]:
                                for rec in reversed(all_step_metrics_storage):
                                    if rec['side'] == other_side and not rec['done']:
                                        raw_val = ((current_frame - rec['start_frame']) / frames_per_second) * 1000
                                        gct_filter_history.append(raw_val)
                                        rec['gct'] = sum(gct_filter_history) / len(gct_filter_history)
                                        rec['done'] = True
                                        mapping = {
                                            'cadence': 'cadence_value (spm)', 'gct': 'gct_value (ms)',
                                            'v_osc': 'vert_osc_value (%)', 'strike_knee': 'knee_strike_angle (deg)',
                                            'push_knee': 'knee_push_angle (deg)', 'elbow': 'elbow_angle_val (deg)',
                                            'split': 'leg_split_val (deg)', 'trunk': 'trunk_lean_value (deg)'
                                        }
                                        step_df = pd.DataFrame([rec]).rename(columns=mapping)
                                        for target, model in models.items():
                                            latest_ai_results[target] = model.predict(step_df[X_features])[0]
                                        leg_is_on_ground[other_side] = False
                                        break

                            leg_is_on_ground[side] = True
                            ankle_y_at_contact[side] = ankle_y
                            last_leg_that_landed = side
                            current_status_event = f"{side.upper()} STRIKE"

                            if previous_strike_frame is not None:
                                cadence_history.append((60 * frames_per_second) / (current_frame - previous_strike_frame))
                                avg_cadence = sum(cadence_history) / len(cadence_history)

                            previous_strike_frame = current_frame
                            last_strike_frame_index = current_frame

                            if side == "Right":
                                right_step_count += 1
                            else:
                                left_step_count += 1

                            ml_data = prepare_ml_features(landmarks, side, fw, fh)
                            all_step_metrics_storage.append({
                                'side': side,
                                'strike_knee': history[mid_idx],
                                'split': current_max_split,
                                'trunk': trunk_angle,
                                'cadence': avg_cadence,
                                'v_osc': v_osc,
                                'elbow': e_ang,
                                'start_frame': current_frame,
                                'push_knee': history[mid_idx],
                                **ml_data,
                                'done': False,
                                'gct': 0
                            })
                            current_max_split = 0

                for side, current_y in [("Right", landmarks[28].y), ("Left", landmarks[27].y)]:
                    if leg_is_on_ground[side]:
                        for rec in reversed(all_step_metrics_storage):
                            if rec['side'] == side and not rec['done']:
                                c_ang = r_knee_ang if side == "Right" else l_knee_ang
                                if c_ang > rec['push_knee']:
                                    rec['push_knee'] = c_ang
                                frames_ground = current_frame - rec['start_frame']
                                moving_up = (current_y < last_ankle_y[side]) if last_ankle_y[side] is not None else False
                                if ((ankle_y_at_contact[side] - current_y) > push_off_threshold and
                                        frames_ground >= min_contact_frames and moving_up):
                                    raw_val = (frames_ground / frames_per_second) * 1000
                                    gct_filter_history.append(raw_val)
                                    rec['gct'] = sum(gct_filter_history) / len(gct_filter_history)
                                    rec['done'] = True
                                    mapping = {
                                        'cadence': 'cadence_value (spm)', 'gct': 'gct_value (ms)',
                                        'v_osc': 'vert_osc_value (%)', 'strike_knee': 'knee_strike_angle (deg)',
                                        'push_knee': 'knee_push_angle (deg)', 'elbow': 'elbow_angle_val (deg)',
                                        'split': 'leg_split_val (deg)', 'trunk': 'trunk_lean_value (deg)'
                                    }
                                    step_df = pd.DataFrame([rec]).rename(columns=mapping)
                                    for target, model in models.items():
                                        latest_ai_results[target] = model.predict(step_df[X_features])[0]
                                    leg_is_on_ground[side] = False
                                    current_status_event = f"{side.upper()} PUSH-OFF"
                                    break
                        last_ankle_y[side] = current_y

                # draw UI panels
                cv2.rectangle(display_frame, (0, 0), (280, 100), (20, 20, 20), -1)
                cv2.putText(display_frame, f"STEPS: {right_step_count + left_step_count}", (15, 35), 1, 1.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"CADENCE: {int(avg_cadence)}", (15, 65), 1, 1.2, (0, 255, 0), 2)
                cv2.putText(display_frame, f"STATUS: {current_status_event}", (15, 95), 1, 1.0, (0, 255, 255), 1)

                if latest_ai_results:
                    bw, start_y, sx = 250, 35, fw - 260
                    cv2.rectangle(display_frame, (sx, 10), (fw - 10, start_y + 160), (20, 20, 20), -1)
                    cv2.putText(display_frame, "AI ANALYSIS", (sx + 15, start_y), 1, 1.2, (255, 255, 255), 2)
                    for i, (metric, score) in enumerate(latest_ai_results.items()):
                        y_p = start_y + 30 + (i * 18)
                        val = int(score)
                        color = (0, 255, 0) if val >= 3 else (0, 255, 255) if val == 2 else (0, 0, 255)
                        cv2.putText(display_frame, f"{metric.replace('_score', '').upper()}:", (sx + 15, y_p), 1, 0.8, (200, 200, 200), 1)
                        cv2.putText(display_frame, f"{val}", (sx + bw - 30, y_p), 1, 1.0, color, 2)

                for side_ui, pos, state in [
                    ("LEFT", (10, fh - 20), leg_is_on_ground["Left"]),
                    ("RIGHT", (fw - 130, fh - 20), leg_is_on_ground["Right"])
                ]:
                    color = (0, 255, 0) if state else (0, 0, 255)
                    cv2.rectangle(display_frame, (pos[0] - 10, pos[1] - 40), (pos[0] + 120, pos[1] + 10), (0, 0, 0), -1)
                    cv2.putText(display_frame, side_ui, (pos[0], pos[1] - 20), 1, 1.2, color, 2)
                    cv2.putText(display_frame, "CONTACT" if state else "FLIGHT", (pos[0], pos[1]), 1, 0.9, (255, 255, 255), 1)

            if writer:
                writer.write(display_frame)
            last_display_frame = display_frame.copy()

    # session summary screen — 3 seconds at end
    if last_display_frame is not None and writer:
        done_steps = [s for s in all_step_metrics_storage if s['done']]
        if done_steps:
            mapping = {
                'cadence': 'cadence_value (spm)', 'gct': 'gct_value (ms)',
                'v_osc': 'vert_osc_value (%)', 'strike_knee': 'knee_strike_angle (deg)',
                'push_knee': 'knee_push_angle (deg)', 'elbow': 'elbow_angle_val (deg)',
                'split': 'leg_split_val (deg)', 'trunk': 'trunk_lean_value (deg)'
            }
            model_df = pd.DataFrame(done_steps).rename(columns=mapping)

            overlay = last_display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, last_display_frame, 0.3, 0, last_display_frame)

            cx, cy = frame_width // 2, frame_height // 2
            cv2.rectangle(last_display_frame, (cx - 250, cy - 225), (cx + 250, cy + 225), (30, 30, 30), -1)
            cv2.rectangle(last_display_frame, (cx - 250, cy - 225), (cx + 250, cy + 225), (255, 255, 255), 2)
            cv2.putText(last_display_frame, "SESSION SUMMARY", (cx - 150, cy - 180), 1, 2.0, (255, 255, 255), 2)

            for i, (target, model) in enumerate(models.items()):
                avg_score = int(round(np.mean(model.predict(model_df[X_features]))))
                color = (0, 255, 0) if avg_score >= 3 else (0, 255, 255) if avg_score == 2 else (0, 0, 255)
                y_p = cy - 100 + (i * 35)
                cv2.putText(last_display_frame, f"{target.replace('_score', '').upper()}:", (cx - 220, y_p), 1, 1.2, (200, 200, 200), 1)
                cv2.putText(last_display_frame, f"{avg_score}", (cx + 180, y_p), 1, 1.5, color, 2)

            for _ in range(int(frames_per_second * 3)):
                writer.write(last_display_frame)

    video_capture.release()
    if writer:
        writer.release()

    done_steps = [s for s in all_step_metrics_storage if s['done']]
    total_steps = right_step_count + left_step_count

    if not done_steps:
        return {"error": "No steps detected. Make sure the video shows a clear side view of someone running.", "total_steps": 0}

    mapping = {
        'cadence': 'cadence_value (spm)', 'gct': 'gct_value (ms)',
        'v_osc': 'vert_osc_value (%)', 'strike_knee': 'knee_strike_angle (deg)',
        'push_knee': 'knee_push_angle (deg)', 'elbow': 'elbow_angle_val (deg)',
        'split': 'leg_split_val (deg)', 'trunk': 'trunk_lean_value (deg)'
    }
    model_df = pd.DataFrame(done_steps).rename(columns=mapping)

    avg_scores = {}
    for target, model in models.items():
        preds = model.predict(model_df[X_features])
        avg_scores[target] = int(round(float(np.mean(preds))))

    avg_metrics = {
        "cadence_spm": round(float(np.mean([s['cadence'] for s in done_steps if s['cadence'] > 0])), 1),
        "gct_ms": round(float(np.mean([s['gct'] for s in done_steps if s['gct'] > 0])), 1),
        "vert_osc_pct": round(float(np.mean([s['v_osc'] for s in done_steps])), 1),
        "trunk_lean_deg": round(float(np.mean([s['trunk'] for s in done_steps])), 1),
    }

    return {
        "total_steps": total_steps,
        "analyzed_steps": len(done_steps),
        "scores": avg_scores,
        "metrics": avg_metrics
    }