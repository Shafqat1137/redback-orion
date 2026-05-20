import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import logging

from config import CONFIG

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================
# LOAD TRACKING DATA
# ============================================================

def load_tracking_data(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = [
        "frame_id",
        "player_id",
        "timestamps_s",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "w",
        "h",
        "confidence"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Tracking CSV is empty.")

    return df


# ============================================================
# DETECT CANDIDATE FRAMES
# ============================================================

def detect_candidate_frames(df):

    detected_frames = []
    grouped = df.groupby("frame_id")

    for frame_id, group in grouped:

        players = len(group)

        if players < CONFIG["MIN_PLAYERS"]:
            continue

        x_min = group["x1"].min()
        y_min = group["y1"].min()
        x_max = group["x2"].max()
        y_max = group["y2"].max()

        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)

        area = width * height

        density_ratio = players / max(area / 100000, 1)

        cluster_score = players + int(density_ratio * 5)

        if (
            cluster_score >= CONFIG["MIN_INTERACTIONS"]
            and density_ratio >= CONFIG["DENSITY_THRESHOLD"]
        ):

            detected_frames.append({
                "frame_id": int(frame_id),
                "timestamps_s": float(group["timestamps_s"].iloc[0]),
                "cluster_score": int(cluster_score),
                "density_ratio": float(density_ratio),
                "players": int(players),
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max)
            })

    return pd.DataFrame(detected_frames)


# ============================================================
# GROUP EVENTS
# ============================================================

def group_events(filtered_df, fps):

    events = []

    if filtered_df.empty:
        return events

    frames = filtered_df["frame_id"].tolist()

    current_start = frames[0]
    previous_frame = frames[0]

    for frame in frames[1:]:

        if frame - previous_frame <= CONFIG["MAX_FRAME_GAP"]:
            previous_frame = frame

        else:

            duration = previous_frame - current_start + 1

            if duration >= CONFIG["MIN_EVENT_LENGTH"]:
                events.append({
                    "start_frame": current_start,
                    "end_frame": previous_frame
                })

            current_start = frame
            previous_frame = frame

    duration = previous_frame - current_start + 1

    if duration >= CONFIG["MIN_EVENT_LENGTH"]:
        events.append({
            "start_frame": current_start,
            "end_frame": previous_frame
        })

    final_events = []

    for idx, event in enumerate(events, start=1):

        start_frame = event["start_frame"]
        end_frame = event["end_frame"]

        duration_frames = end_frame - start_frame + 1

        final_events.append({
            "event_id": idx,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_time_s": round(start_frame / fps, 3),
            "end_time_s": round(end_frame / fps, 3),
            "duration_frames": int(duration_frames),
            "duration_s": round(duration_frames / fps, 3),
            "label": "tackle"
        })

    return final_events


# ============================================================
# EXTRACT PLAYERS
# ============================================================

def extract_players_for_event(df, start_frame, end_frame):

    event_df = df[
        (df["frame_id"] >= start_frame) &
        (df["frame_id"] <= end_frame)
    ]

    players_output = []

    for player_id in event_df["player_id"].unique():

        player_df = event_df[event_df["player_id"] == player_id]

        avg_bbox = {
            "x1": int(player_df["x1"].mean()),
            "y1": int(player_df["y1"].mean()),
            "x2": int(player_df["x2"].mean()),
            "y2": int(player_df["y2"].mean())
        }

        players_output.append({
            "player_id": int(player_id),
            "avg_bbox": avg_bbox,
            "frames_present": player_df["frame_id"].astype(int).tolist()
        })

    return players_output


# ============================================================
# BUILD JSON
# ============================================================

def build_json(events, df, filtered_df, video_name, fps):

    final_events = []

    for event in events:

        start_frame = event["start_frame"]
        end_frame = event["end_frame"]

        frame_rows = filtered_df[
            (filtered_df["frame_id"] >= start_frame) &
            (filtered_df["frame_id"] <= end_frame)
        ]

        if frame_rows.empty:
            continue

        tackle_bbox = {
            "x_min": int(frame_rows["x_min"].min()),
            "y_min": int(frame_rows["y_min"].min()),
            "x_max": int(frame_rows["x_max"].max()),
            "y_max": int(frame_rows["y_max"].max())
        }

        players_involved = extract_players_for_event(
            df, start_frame, end_frame
        )

        event_output = {
            **event,
            "cluster_score": int(frame_rows["cluster_score"].max()),
            "density_ratio": round(float(frame_rows["density_ratio"].max()), 3),
            "players_detected": len(players_involved),
            "tackle_bbox": tackle_bbox,
            "players_involved": players_involved
        }

        final_events.append(event_output)

    return {
        "video_name": video_name,
        "fps": fps,
        "total_detected_events": len(final_events),
        "events": final_events
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"JSON saved: {path}")


def save_csv(data, path):

    rows = []

    for e in data["events"]:
        rows.append({
            "event_id": e["event_id"],
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "start_time_s": e["start_time_s"],
            "end_time_s": e["end_time_s"],
            "players_detected": e["players_detected"],
            "cluster_score": e["cluster_score"],
            "density_ratio": e["density_ratio"]
        })

    pd.DataFrame(rows).to_csv(path, index=False)

    logging.info(f"CSV saved: {path}")


# ============================================================
# VIDEO GENERATION
# ============================================================

def generate_output_video(video_path, events_json, output_video):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        for event in events_json["events"]:

            if event["start_frame"] <= frame_id <= event["end_frame"]:

                b = event["tackle_bbox"]

                cv2.rectangle(
                    frame,
                    (b["x_min"], b["y_min"]),
                    (b["x_max"], b["y_max"]),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    f"TACKLE #{event['event_id']}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Players: {event['players_detected']}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    logging.info(f"Video saved: {output_video}")


# ============================================================
# MAIN CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--tracking", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=29.97)

    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    logging.info("Loading tracking data...")
    df = load_tracking_data(args.tracking)

    logging.info("Detecting frames...")
    filtered_df = detect_candidate_frames(df)

    logging.info(f"Detected frames: {len(filtered_df)}")

    logging.info("Grouping events...")
    events = group_events(filtered_df, args.fps)

    logging.info(f"Events: {len(events)}")

    logging.info("Building JSON...")
    final_json = build_json(
        events, df, filtered_df,
        Path(args.video).name,
        args.fps
    )

    save_json(final_json, out / "tackle_events.json")
    save_csv(final_json, out / "tackle_events.csv")

    logging.info("Generating video...")
    generate_output_video(
        args.video,
        final_json,
        out / "tackle_detection_output.mp4"
    )

    logging.info("Pipeline completed")


if __name__ == "__main__":
    main()