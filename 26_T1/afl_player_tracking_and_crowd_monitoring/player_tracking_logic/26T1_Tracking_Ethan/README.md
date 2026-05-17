# YOLO Player Tracking Pipeline

This project provides a modular YOLO-based player tracking pipeline for AFL match footage. It detects players in each video frame, assigns persistent tracking IDs, exports an annotated tracking video, and writes a backend-readable JSON file containing the full tracking history for each ID.

The pipeline is designed for cases where simple centre-distance tracking is not stable enough, especially in AFL broadcast footage where players frequently overlap, move quickly, appear in visually similar groups, and may be detected with fluctuating class predictions.


## Latest update

This version has been updated for the latest `best(SSvsWB).pt` model and now integrates a JerseyColorDetection-style post-processing stage. In addition to player detection and ID tracking, the pipeline can extract torso-region jersey colour features, cluster tracked players into colour-based groups, and export movement metrics such as approximate distance and speed.

For the current `best(SSvsWB).pt` model, the class mapping is:

```text
0 = REF
1 = SS
2 = WB
```

For player-only tracking, set `player_classes` to `[1, 2]`. If referee tracking is also required, use `[0, 1, 2]`.

## Key features

- YOLO-based player detection using a custom model.
- Persistent ID assignment across frames.
- Initial class preservation for every tracking ID.
- Robust multi-factor detection-to-track matching.
- Short-term trajectory consistency scoring.
- Bounding-box IoU comparison.
- Jersey colour / lightweight appearance similarity.
- Duplicate detection suppression.
- Ambiguous match handling to reduce unnecessary ID switches.
- Annotated output video with ID labels, confidence values, bounding boxes, and motion trails.
- JSON output for backend integration and future tracking-data processing.
- Latest custom model support for `best(SSvsWB).pt`.
- JerseyColorDetection-style torso colour feature extraction.
- Player-level KMeans clustering using jersey colour features.
- Cluster labels added to the annotated video and JSON output.
- CSV movement metrics output including approximate distance and speed.

## Project structure

```text
player_tracking_project/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   └── videos/
│       └── video.mp4
├── models/
│   └── model.pt
└── player_tracking/
    ├── __init__.py
    ├── appearance.py
    ├── config.py
    ├── detector.py
    ├── geometry.py
    ├── jersey_cluster.py
    ├── metrics.py
    ├── tracker.py
    ├── video_processor.py
    └── visualizer.py
```

### Main modules

| File | Purpose |
|---|---|
| `main.py` | Command-line entry point. Reads runtime arguments and creates the tracking configuration. |
| `config.py` | Stores all detection, tracking, matching, appearance, and output parameters. |
| `video_processor.py` | Loads the YOLO model and video, processes frames, writes the output video, and exports JSON. |
| `detector.py` | Converts YOLO results into clean detection records and removes likely duplicate detections. |
| `tracker.py` | Maintains active tracks, performs robust ID matching, handles missing tracks, and exports track records. |
| `geometry.py` | Provides distance, IoU, cosine similarity, direction cost, and bbox helper functions. |
| `appearance.py` | Extracts lightweight HSV colour histogram features from each detected player crop. |
| `jersey_cluster.py` | Assigns player-level jersey colour clusters using StandardScaler and KMeans after tracking. |
| `metrics.py` | Exports per-track movement metrics such as approximate distance and speed to CSV. |
| `visualizer.py` | Draws bounding boxes, ID labels, confidence values, centre points, and trajectory trails. |

## Installation

Create and activate a Python environment:

```bash
conda create -n player_tracking python=3.11 -y
conda activate player_tracking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```text
ultralytics
opencv-python
numpy
scipy
scikit-learn
```

`scipy` is used for Hungarian assignment through `linear_sum_assignment`. If it is unavailable, the tracker falls back to a greedy assignment strategy.

`scikit-learn` is required for the JerseyColorDetection-style clustering stage, where `StandardScaler` and `KMeans` are used to group tracked players based on jersey colour features.

## Prepare input files

Place the input video and YOLO model in the following default locations:

```text
data/videos/video.mp4
models/model.pt
```

The default paths can also be changed through command-line arguments.


For the latest SS vs WB test, place the updated model at:

```text
models/best(SSvsWB).pt
```

The current tested video can still be placed at:

```text
data/videos/video.mp4
```

## Run the tracker

Default run:

```bash
python main.py
```

Custom run:

```bash
python main.py \
  --video data/videos/video.mp4 \
  --model models/model.pt \
  --output-video video_track.mp4 \
  --output-json video_track.json \
  --seconds 40 \
  --conf 0.25 \
  --imgsz 640
```


Latest SS vs WB run with jersey clustering:

```bash
python main.py \
  --video data/videos/video.mp4 \
  --model "models/best(SSvsWB).pt" \
  --output-video outputs/video_track_clustered.mp4 \
  --output-json outputs/video_track_clustered.json \
  --output-csv outputs/player_metrics.csv \
  --seconds 40 \
  --conf 0.10 \
  --imgsz 960 \
  --clusters 2
```

Use `--clusters 2` when only SS and WB are tracked. Use `--clusters 3` if REF is also included in `player_classes`.

### Command-line arguments

| Argument | Default | Description |
|---|---:|---|
| `--video` | `data/videos/video.mp4` | Input video path. |
| `--model` | `models/model.pt` | YOLO model path. |
| `--output-video` | `video_track.mp4` | Output annotated video path. |
| `--output-json` | `video_track.json` | Output JSON path. |
| `--output-csv` | `outputs/player_metrics.csv` | Output CSV path for movement metrics. |
| `--seconds` | `40` | Number of seconds to process from the beginning of the video. |
| `--conf` | `0.25` | YOLO confidence threshold. |
| `--imgsz` | `640` | YOLO inference image size. |
| `--clusters` | `3` | Number of jersey colour clusters used in KMeans. Use `2` for two player teams only. |
| `--min-samples-per-track` | `5` | Minimum number of jersey colour samples required before a track can be clustered. |
| `--pixel-to-meter` | `0.05` | Approximate conversion factor used when exporting distance and speed metrics. |

## Tracking stability design

The tracker is built around a robust detection-to-track association process. Instead of assigning IDs only by the nearest centre point, each candidate match is evaluated using several complementary signals.

### 1. Initial class preservation

Each new track stores the class predicted by YOLO when the ID is first created:

```text
initial_class_id
initial_class_name
```

This initial class is then kept as the stable class label for that ID. This is useful when YOLO occasionally switches a player between class `0` and class `1` in later frames. The raw per-frame class predictions are still preserved in the JSON output, so the backend can inspect class changes if required.

### 2. Robust ID matching

For every active track and every new detection, the tracker builds a matching cost. A lower cost means the detection is more likely to belong to that track.

The total cost combines:

```text
distance cost
+ IoU cost
+ direction cost
+ appearance cost
```

In `config.py`, the default weighted cost is:

```text
total_cost =
    0.45 * distance_cost
  + 0.25 * iou_cost
  + 0.15 * direction_cost
  + 0.15 * appearance_cost
```

The weights can be tuned through:

```python
distance_weight = 0.45
iou_weight = 0.25
direction_weight = 0.15
appearance_weight = 0.15
```

### 3. Centre-distance gating

The tracker compares the current detection centre with the last known centre of each active track. This provides a basic motion constraint and prevents unrealistic jumps.

The key parameter is:

```python
max_distance = 120.0
```

Because distance is only one part of the matching cost, this value can be more tolerant than in a pure distance-only tracker.

### 4. Trajectory prediction through direction consistency

The tracker uses the previous centre and current centre of a track to estimate its recent movement direction. A candidate detection is preferred when it continues the same general direction.

This helps stabilise IDs when players are moving quickly or crossing near each other. It is a lightweight trajectory-consistency method rather than a full Kalman filter or deep ReID tracker.

### 5. IoU-based bounding-box comparison

The tracker compares the previous bounding box of a track with each candidate detection using Intersection over Union (IoU).

A high IoU indicates that the detection spatially overlaps with the previous track location, which is useful when players move gradually between frames. If a detection is far away and has almost no IoU overlap, the tracker adds a soft penalty to discourage an unsafe assignment.

### 6. Jersey colour / appearance similarity

For each detection, the pipeline extracts a compact HSV colour histogram from the player crop. This acts as a lightweight appearance descriptor.

The tracker compares appearance features using cosine similarity. This helps reduce ID switches when two players are close together but have different jersey colours or visual appearance.

The appearance model is intentionally lightweight and dependency-friendly. It should be treated as a stabilisation cue, not as a full person re-identification model.


Latest update: the appearance feature is now extracted from the torso / jersey region rather than the full bounding box. This reduces noise from grass, shorts, socks, field markings, and background pixels. The same torso region is also used to create a separate JerseyColorDetection-style feature vector for team clustering.

### 7. Duplicate detection filtering

YOLO may sometimes generate multiple boxes for the same player. Before matching, detections are sorted by confidence and likely duplicates are removed.

A detection is treated as a duplicate when either:

- its centre is too close to a higher-confidence detection, or
- its IoU overlap with a higher-confidence detection is too large.

Key parameter:

```python
duplicate_center_distance = 35.0
```

This reduces cases where one real player receives multiple IDs in the same frame.

### 8. Ambiguous match handling

When the best match and second-best match are too similar, the assignment is considered uncertain. In that case, the tracker avoids forcing a potentially wrong update.

Key parameters:

```python
match_cost_threshold = 0.85
ambiguity_margin = 0.08
confident_cost_threshold = 0.45
```

This is especially useful during overlap or occlusion, where forcing a match can easily cause ID switches.

### 9. New-track suppression near active tracks

If an unmatched detection is close to an existing active track, the tracker does not immediately create a new ID. This helps avoid duplicate IDs when a player is briefly matched ambiguously or partially occluded.

Key parameter:

```python
new_track_suppression_distance = 60.0
```

### 10. Missing-track tolerance

A track is not deleted immediately when it is missed in a frame. Instead, it remains active for a limited number of frames.

Key parameter:

```python
max_missing = 15
```

This helps maintain ID continuity when a player is briefly hidden, missed by the detector, or affected by motion blur.


### 11. JerseyColorDetection-style team clustering

After tracking is complete, the pipeline collects jersey colour samples for each tracking ID and performs player-level clustering. This is different from frame-by-frame team classification because it uses all available samples from the same ID to build a more stable colour profile.

The clustering process uses:

```text
torso ROI extraction
HSV colour statistics
red / yellow / blue / white / dark pixel ratios
median feature aggregation per track
StandardScaler feature normalisation
KMeans unsupervised clustering
```

The exported cluster labels are stored as:

```text
cluster_id
cluster_team
```

These labels are useful for separating SS and WB players in the output video and backend JSON. The cluster names are automatic labels such as `Team_0` and `Team_1`; they should be mapped to real team names after visually checking the output video.

## Algorithm overview

For each frame:

1. Read the next video frame.
2. Run YOLO inference.
3. Keep only configured player classes, such as class `0` and class `1`.
4. Convert YOLO boxes into detection records:
   - bounding box
   - centre point
   - confidence
   - detected class
   - class name
   - appearance feature
5. Remove likely duplicate detections.
6. Build a cost matrix between active tracks and current detections.
7. Match tracks to detections using Hungarian assignment when available.
8. Reject matches whose cost is too high.
9. Reject ambiguous matches when the best and second-best candidates are too close.
10. Update matched tracks.
11. Mark unmatched tracks as missing.
12. Create new tracks only when unmatched detections are not too close to existing active tracks.
13. Remove tracks that have been missing for longer than `max_missing`.
14. Draw boxes, IDs, confidence values, and trails on the output video frame.
15. Save all tracking history to the final JSON file.


Latest update after the frame loop:

16. Collect jersey colour samples from each tracking ID.
17. Compute a median representative jersey feature for each player track.
18. Use StandardScaler and KMeans to assign colour-based `cluster_team` labels.
19. Render the final annotated video with ID labels, confidence values, trails, and cluster labels.
20. Export a CSV file containing per-track movement metrics such as approximate distance and speed.
21. Save the final JSON file with tracking history, cluster labels, and jersey feature summaries.

## Configuration parameters

Most tracking behaviour can be tuned in:

```text
player_tracking/config.py
```

### Detection parameters

| Parameter | Default | Description |
|---|---:|---|
| `player_classes` | `[0, 1]` | YOLO classes treated as players for tracking. |
| `conf_threshold` | `0.25` | Minimum YOLO detection confidence. |
| `imgsz` | `640` | YOLO inference image size. |
| `process_seconds` | `40` | Number of seconds processed from the video. |

For `best(SSvsWB).pt`, use `[1, 2]` for SS and WB player-only tracking, because the model classes are `0 = REF`, `1 = SS`, and `2 = WB`.

### Tracking parameters

| Parameter | Default | Description |
|---|---:|---|
| `max_distance` | `120.0` | Maximum expected movement distance used in matching cost. |
| `trail_length` | `30` | Number of recent centre points drawn as the trajectory trail. |
| `max_missing` | `15` | Number of frames a track may be missing before removal. |
| `duplicate_center_distance` | `35.0` | Distance threshold for suppressing duplicate detections. |
| `new_track_suppression_distance` | `60.0` | Prevents duplicate ID creation near active tracks. |

### Matching cost parameters

| Parameter | Default | Description |
|---|---:|---|
| `distance_weight` | `0.45` | Weight for centre-distance matching. |
| `iou_weight` | `0.25` | Weight for bounding-box IoU matching. |
| `direction_weight` | `0.15` | Weight for trajectory direction consistency. |
| `appearance_weight` | `0.15` | Weight for jersey colour / appearance similarity. |
| `match_cost_threshold` | `0.85` | Maximum accepted matching cost. |
| `ambiguity_margin` | `0.08` | Minimum separation required between best and second-best matches. |
| `confident_cost_threshold` | `0.45` | Matches below this cost are treated as confident. |

### Appearance parameters

| Parameter | Default | Description |
|---|---:|---|
| `appearance_momentum` | `0.8` | Smooths the appearance feature over time. Higher values preserve older appearance more strongly. |


### Jersey clustering parameters

| Parameter | Default | Description |
|---|---:|---|
| `jersey_clusters` | `3` | Number of KMeans clusters for jersey colour grouping. Use `2` for SS/WB only, or `3` when REF is included. |
| `min_samples_per_track` | `5` | Minimum number of torso colour samples needed before a track can be clustered. |
| `kmeans_random_state` | `42` | Random seed for reproducible KMeans clustering. |

### Metrics parameters

| Parameter | Default | Description |
|---|---:|---|
| `pixel_to_meter` | `0.05` | Approximate pixel-to-metre conversion used for distance and speed output. |
| `max_speed_kmh` | `40.0` | Filters unrealistic peak speed values when calculating movement metrics. |

### Output parameters

| Parameter | Default | Description |
|---|---:|---|
| `draw_confidence` | `True` | Draws the latest confidence score beside each ID. |
| `draw_cluster` | `True` | Draws the assigned jersey colour cluster label beside each ID. |
| `print_every_n_frames` | `50` | Prints progress every N processed frames. |

## Outputs

The pipeline writes two default output files:

```text
video_track.mp4
video_track.json
```


The latest JerseyColorDetection-style version can also write:

```text
outputs/video_track_clustered.mp4
outputs/video_track_clustered.json
outputs/player_metrics.csv
```

### Annotated video

The output video contains:

- tracking ID
- fixed initial class name
- current confidence score
- bounding box
- centre point
- recent trajectory trail
- jersey colour cluster label such as `Team_0` or `Team_1`

### JSON output

The JSON file is intended to be readable by the backend team and usable as a reference structure for future integration.

Top-level fields include:

| Field | Description |
|---|---|
| `video_path` | Input video path. |
| `model_path` | YOLO model path. |
| `output_video_path` | Output tracking video path. |
| `output_json_path` | Output JSON path. |
| `output_csv_path` | Output CSV metrics path. |
| `fps` | Video frame rate. |
| `total_frames` | Total number of frames in the input video. |
| `processed_frames` | Number of frames processed by the pipeline. |
| `process_seconds` | Configured processing duration. |
| `resolution` | Video width and height. |
| `player_classes` | Classes tracked as players. |
| `jersey_clustering` | Summary of the KMeans jersey clustering stage. |
| `tracks` | Dictionary of all tracking IDs and their histories. |

Each track contains:

| Field | Description |
|---|---|
| `track_id` | Persistent tracking ID. |
| `initial_class_id` | Class assigned when the ID was first created. |
| `initial_class_name` | Class name assigned when the ID was first created. |
| `num_observed_frames` | Total number of frames where this ID was detected and updated. |
| `cluster_id` | Numeric jersey colour cluster assigned after KMeans clustering. |
| `cluster_team` | Human-readable cluster label such as `Team_0`, `Team_1`, or `Unknown`. |
| `jersey_sample_count` | Number of torso colour samples collected for this track. |
| `representative_jersey_feature` | Median jersey colour feature vector for this track. |
| `frames` | Per-frame tracking history for this ID. |

Each frame record contains:

| Field | Description |
|---|---|
| `frame_index` | Frame number in the processed video. |
| `time_sec` | Timestamp in seconds. |
| `bbox` | Bounding box in `[x1, y1, x2, y2]` format. |
| `center` | Bounding-box centre point in `[x, y]` format. |
| `confidence` | YOLO confidence score for the matched detection. |
| `detected_class_id` | Raw YOLO class prediction in this frame. |
| `detected_class_name` | Raw YOLO class name in this frame. |
| `fixed_initial_class_id` | Stable initial class ID for this track. |
| `fixed_initial_class_name` | Stable initial class name for this track. |
| `cluster_id` | Jersey colour cluster assigned to this track. |
| `cluster_team` | Cluster label assigned to this track. |

Example structure:

```json
{
  "video_path": "data/videos/video.mp4",
  "model_path": "models/model.pt",
  "output_video_path": "video_track.mp4",
  "fps": 30.0,
  "total_frames": 3000,
  "processed_frames": 1200,
  "process_seconds": 40,
  "resolution": {
    "width": 1920,
    "height": 1080
  },
  "player_classes": [0, 1],
  "tracks": {
    "0": {
      "track_id": 0,
      "initial_class_id": 0,
      "initial_class_name": "class_0",
      "num_observed_frames": 52,
      "frames": [
        {
          "frame_index": 0,
          "time_sec": 0.0,
          "bbox": [650.5, 320.1, 703.8, 421.6],
          "center": [677.15, 370.85],
          "confidence": 0.86,
          "detected_class_id": 0,
          "detected_class_name": "class_0",
          "fixed_initial_class_id": 0,
          "fixed_initial_class_name": "class_0"
        }
      ]
    }
  }
}
```


### CSV metrics output

The `player_metrics.csv` file provides one row per tracking ID. It includes:

| Field | Description |
|---|---|
| `track_id` | Persistent tracking ID. |
| `cluster_id` | Jersey colour cluster ID. |
| `cluster_team` | Jersey colour cluster label. |
| `initial_class_id` | Initial YOLO class ID. |
| `initial_class_name` | Initial YOLO class name. |
| `num_observed_frames` | Number of frames where the track was observed. |
| `jersey_sample_count` | Number of jersey colour samples used for clustering. |
| `first_frame` / `last_frame` | First and last frame indices for the track. |
| `first_time_sec` / `last_time_sec` | First and last timestamps for the track. |
| `total_distance_px` | Approximate total movement distance in pixels. |
| `total_distance_m` | Approximate total movement distance after applying `pixel_to_meter`. |
| `avg_speed_kmh` | Approximate average speed. |
| `max_speed_kmh` | Approximate maximum speed after filtering unrealistic values. |

Distance and speed are approximate because they are calculated from image-space coordinates. For more accurate tactical analysis, player positions should be mapped to a stable field coordinate system using homography or camera registration.

## Backend integration notes

The JSON structure can support backend-side analysis such as:

- retrieving all frames associated with a tracking ID
- reconstructing bounding-box history
- reconstructing confidence history
- reconstructing position history
- calculating first and last detected frame
- calculating first and last detected timestamp
- checking raw class prediction changes over time
- measuring how often an ID appears
- generating trajectory data for visualisation or analytics
- grouping players by jersey colour cluster for team-based analysis
- using `player_metrics.csv` for approximate movement distance and speed analysis

The current implementation stores the bbox centre as `center`. If the backend requires the bottom-centre point for player-ground-position estimation, it can be derived from each `bbox` as:

```python
bottom_center_x = (x1 + x2) / 2
bottom_center_y = y2
```

First and last detection information can be derived from the first and last entries in each track's `frames` list:

```python
first_frame = track["frames"][0]["frame_index"]
first_time = track["frames"][0]["time_sec"]

last_frame = track["frames"][-1]["frame_index"]
last_time = track["frames"][-1]["time_sec"]
```

Raw class prediction history can be derived from:

```python
class_history = [frame["detected_class_id"] for frame in track["frames"]]
```

Confidence history can be derived from:

```python
confidence_history = [frame["confidence"] for frame in track["frames"]]
```

Bounding-box history can be derived from:

```python
bbox_history = [frame["bbox"] for frame in track["frames"]]
```

Centre-position history can be derived from:

```python
center_history = [frame["center"] for frame in track["frames"]]
```


Cluster labels can be accessed from:

```python
cluster_team = track["cluster_team"]
cluster_id = track["cluster_id"]
```

## Practical tuning advice

For dense AFL scenes, the following parameters are usually the most important:

- Increase `max_missing` if players are frequently lost during short occlusions.
- Increase `new_track_suppression_distance` if duplicate IDs appear near the same player.
- Increase `appearance_weight` if jersey colour is helpful for separating nearby players.
- Increase `direction_weight` if IDs switch during fast but consistent movement.
- Increase `iou_weight` if the camera is stable and boxes overlap strongly between frames.
- Decrease `match_cost_threshold` if the tracker accepts too many unsafe matches.
- Increase `ambiguity_margin` if ID switches occur during overlap.
- Increase `conf_threshold` if there are too many false detections.
- Decrease `conf_threshold` if players are frequently missed.
- Use `player_classes = [1, 2]` for the latest `best(SSvsWB).pt` model when only SS and WB players should be tracked.
- Use `--clusters 2` for two-team clustering, or `--clusters 3` if referees are included.
- Increase `min_samples_per_track` if unstable or short-lived tracks are being clustered too easily.
- Tune `pixel_to_meter` before treating CSV distance and speed values as real-world measurements.

## Limitations

This tracker is designed as a practical lightweight pipeline, not a full deep re-identification system. The appearance feature is based on HSV colour histograms, so it may be less reliable when:

- both teams have visually similar jerseys
- players are very small in the frame
- lighting changes significantly
- motion blur reduces crop quality
- heavy occlusion hides most of the player body
- broadcast camera movement changes the visual context quickly
- KMeans cluster labels are automatic and need to be mapped to real team names manually
- distance and speed values are approximate because they are currently calculated from image-space coordinates

For stronger long-term identity preservation, future work could add camera-motion compensation, optical flow, Kalman filtering, or a trained ReID feature extractor.

## Recommended output for team handover

For sharing results with the backend team, provide:

```text
video_track.mp4
video_track.json
```

For the latest JerseyColorDetection-style version, also provide:

```text
outputs/video_track_clustered.mp4
outputs/video_track_clustered.json
outputs/player_metrics.csv
```

The video is useful for visual inspection, while the JSON file is the main structured output for backend integration.
