import cv2
import numpy as np
import json
import csv
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# --- CONFIGURATION ---
VIDEO_PATH = "../afl_video.mp4"
JSON_INPUT_PATH = "../afl_video_tracking.json"
OUTPUT_VIDEO_PATH = "annotated_output.mp4"
OUTPUT_JSON_PATH = "final_results.json"
OUTPUT_CSV_PATH = "final_results.csv"

def extract_features(frame, bbox_dict):
    try:
        x1, y1, x2, y2 = int(bbox_dict['x1']), int(bbox_dict['y1']), int(bbox_dict['x2']), int(bbox_dict['y2'])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return [np.median(hsv[:,:,0]), np.median(hsv[:,:,1]), np.median(hsv[:,:,2])]
    except: return None

# Initialize
# Force FFmpeg backend to avoid GStreamer warnings on Windows
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
out_video = None

try:
    # --- STAGE 1: JSON & VIDEO VERIFICATION ---
    print(f"--- DEBUG STAGE 1: INITIALIZATION ---")
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
        sys.exit()
    
    with open(JSON_INPUT_PATH, 'r') as f:
        full_data = json.load(f)
        tracking_results = full_data.get('tracking_results', [])
    
    print(f"[INFO] JSON loaded. Found {len(tracking_results)} frames in tracking data.")
    frame_map = {d['frame_number']: d['players'] for d in tracking_results}
    
    # --- STAGE 2: FEATURE COLLECTION ---
    print(f"\n--- DEBUG STAGE 2: FEATURE COLLECTION ---")
    player_observations = defaultdict(list)
    curr_frame = 1
    frames_processed = 0
    detections_found = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print(f"[INFO] Video stream ended at frame {curr_frame-1}")
            break
        
        if curr_frame in frame_map:
            players = frame_map[curr_frame]
            for p in players:
                feat = extract_features(frame, p['bbox'])
                if feat:
                    player_observations[p['player_id']].append(feat)
                    detections_found += 1
            frames_processed += 1

        if curr_frame % 500 == 0:
            print(f"  > Progress: Frame {curr_frame} | Detections cached: {detections_found}")
        curr_frame += 1

    print(f"[RESULT] Processed {frames_processed} frames. Unique IDs found: {len(player_observations)}")

    # --- STAGE 3: CLUSTERING ANALYSIS ---
    print(f"\n--- DEBUG STAGE 3: CLUSTERING ---")
    valid_ids, id_fingerprints = [], []
    for tid, feats in player_observations.items():
        if len(feats) > 0:
            id_fingerprints.append(np.mean(feats, axis=0))
            valid_ids.append(tid)

    if not id_fingerprints:
        print("[ERROR] No valid player features were extracted. Clustering aborted.")
        sys.exit()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(id_fingerprints)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20).fit(scaled_data)
    id_to_team = {valid_ids[i]: int(kmeans.labels_[i]) for i in range(len(valid_ids))}
    print(f"[RESULT] Clustering successful. Team distribution: {np.bincount(kmeans.labels_)}")

    # --- STAGE 4: WRITER INITIALIZATION ---
    print(f"\n--- DEBUG STAGE 4: OUTPUT WRITING ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Use MP4V with forced FFMPEG
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    if not out_video.isOpened():
        print("[ERROR] VideoWriter failed to open. Check write permissions or codec.")
    
    final_json_data = []
    curr_frame = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_entry = {"frame": curr_frame, "players": []}
        if curr_frame in frame_map:
            for p in frame_map[curr_frame]:
                tid = p['player_id']
                team_id = id_to_team.get(tid, -1)
                # Drawing logic...
                frame_entry["players"].append({"id": tid, "team": team_id})
        
        out_video.write(frame)
        final_json_data.append(frame_entry)
        curr_frame += 1

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_json_data, f)
    print(f"[RESULT] Final JSON saved with {len(final_json_data)} entries.")

finally:
    if cap: cap.release()
    if out_video: out_video.release()
    print("\n--- DEBUG: PROCESS COMPLETE ---")