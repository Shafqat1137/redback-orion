import cv2
import numpy as np
import json
import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# --- CONFIGURATION ---
VIDEO_PATH = "../afl_video.mp4"
JSON_INPUT_PATH = "../afl_video_tracking.json"
OUTPUT_VIDEO_PATH = "annotated_output.mp4"
OUTPUT_JSON_PATH = "final_results.json"
OUTPUT_CSV_PATH = "final_results.csv"

N_CLUSTERS = 3 

def extract_features(frame, bbox_dict):
    """Extracts features using the dictionary-style bbox from your JSON."""
    x1, y1 = int(bbox_dict['x1']), int(bbox_dict['y1'])
    x2, y2 = int(bbox_dict['x2']), int(bbox_dict['y2'])
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 10: return None

    # ROI for Torso
    roi = frame[y1 + int(bh*0.2):y1 + int(bh*0.55), x1 + int(bw*0.25):x1 + int(bw*0.75)]
    if roi.size == 0: return None
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    def get_ratio(img, low, high):
        mask = cv2.inRange(img, np.array(low), np.array(high))
        return cv2.countNonZero(mask) / img.size

    return [
        np.median(hsv[:,:,0]), np.median(hsv[:,:,1]), np.median(hsv[:,:,2]),
        get_ratio(hsv, [0, 70, 50], [10, 255, 255]) + get_ratio(hsv, [170, 70, 50], [180, 255, 255]),
        get_ratio(hsv, [90, 50, 50], [130, 255, 255]),
        get_ratio(hsv, [0, 0, 160], [180, 50, 255]),
        get_ratio(hsv, [0, 0, 0], [180, 255, 60])
    ]

# --- PHASE 1: DATA LOADING & FEATURE COLLECTION ---
print("[1/3] Collecting global player features...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

with open(JSON_INPUT_PATH, 'r') as f:
    full_data = json.load(f)
    # Correcting the access point based on your JSON example
    tracking_results = full_data.get('tracking_results', [])

player_observations = defaultdict(list)

for frame_data in tracking_results:
    # frame_number in your JSON starts at 1, so we skip frames accordingly
    target_frame = frame_data['frame_number'] - 1 
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret: continue
    
    players = frame_data.get('players', [])
    for p in players:
        tid = p['player_id']
        bbox = p['bbox']
        feat = extract_features(frame, bbox)
        if feat:
            player_observations[tid].append(feat)

# --- PHASE 2: GLOBAL CLUSTERING ---
print("[2/3] Performing global K-Means clustering...")
valid_ids, id_fingerprints = [], []

for tid, feats in player_observations.items():
    if len(feats) > 5:
        id_fingerprints.append(np.mean(feats, axis=0))
        valid_ids.append(tid)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(id_fingerprints)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
clusters = kmeans.fit_predict(scaled_data)
id_to_team = {valid_ids[i]: int(clusters[i]) for i in range(len(valid_ids))}

# --- PHASE 3: OUTPUT GENERATION ---
print("[3/3] Generating final video and data files...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                           (int(cap.get(3)), int(cap.get(4))))

TEAM_COLORS = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 255)}
final_json_data = []
csv_rows = []

for frame_data in tracking_results:
    target_frame = frame_data['frame_number'] - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret: break
    
    current_frame_output = {"frame": frame_data['frame_number'], "players": []}
    
    for p in frame_data.get('players', []):
        tid = p['player_id']
        team_id = id_to_team.get(tid, -1)
        team_name = f"Team_{team_id}" if team_id != -1 else "Unknown"
        
        color = TEAM_COLORS.get(team_id, (128, 128, 128))
        box = p['bbox']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        
        # Annotate
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid} {team_name}", (x1, y1-10), 0, 0.5, color, 2)
        
        current_frame_output["players"].append({"player_id": tid, "team": team_name, "bbox": box})
        csv_rows.append([frame_data['frame_number'], tid, team_name, x1, y1, x2, y2])
    
    out_video.write(frame)
    final_json_data.append(current_frame_output)

# Export
with open(OUTPUT_JSON_PATH, 'w') as f: json.dump(final_json_data, f, indent=4)
with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "player_id", "team", "x1", "y1", "x2", "y2"])
    writer.writerows(csv_rows)

cap.release()
out_video.release()
print("Done!")