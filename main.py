# main.py
import cv2
import torch
import torchreid
import numpy as np
import json
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- 1. CONFIGURATION ---

# Input and Output paths
INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
BROADCAST_VIDEO = os.path.join(INPUT_DIR, "broadcast.mp4")
TACTICAM_VIDEO = os.path.join(INPUT_DIR, "tacticam.mp4")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODEL AND MATCHING PARAMETERS ---
YOLO_MODEL_PATH = "best.pt"
USE_OSNET = True
REID_WEIGHTS = 'facebookresearch/WSL-Images'
REID_MODEL_NAME = 'resnext101_32x8d_wsl'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HOMOGRAPHY AND MATCHING WEIGHTS ---
SPATIAL_DISTANCE_THRESHOLD = 150.0
MATCH_COST_THRESHOLD = 0.7

# --- 2. CORE FUNCTIONS ---

def extract_color_histogram(cropped_img):
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def track_players_in_video(video_path, model):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    all_tracks = {}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, classes=[0], verbose=False)[0]
        detections = [
            ([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(conf), 0)
            for x1, y1, x2, y2, conf, cls in results.boxes.data.tolist()
        ]
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed(): continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            if track_id not in all_tracks:
                all_tracks[track_id] = {"boxes": []}
            all_tracks[track_id]["boxes"].append([frame_idx] + list(map(int, ltrb)))
        frame_idx += 1
    cap.release()
    return all_tracks

def extract_reid_features(video_path, tracks_data, reid_model):
    print(f"Extracting Re-ID features from {video_path}")
    cap = cv2.VideoCapture(video_path)
    reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for track_id, data in tqdm(tracks_data.items()):
        all_features = []
        all_colors = []
        for frame_num, x1, y1, x2, y2 in data["boxes"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret: continue
            player_img = frame[y1:y2, x1:x2]
            if player_img.size == 0: continue
            pil_img = Image.fromarray(cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB))
            img_tensor = reid_transform(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feature = reid_model(img_tensor).cpu().numpy().flatten()
            all_features.append(feature)

            jersey_crop = frame[y1:(y1 + y2)//2, x1:x2]
            if jersey_crop.size > 0:
                hist = extract_color_histogram(jersey_crop)
                all_colors.append(hist)

        if all_features:
            avg_feat = np.mean(all_features, axis=0)
            tracks_data[track_id]['avg_feature'] = avg_feat.tolist()
        if all_colors:
            avg_color = np.mean(all_colors, axis=0)
            tracks_data[track_id]['color_feature'] = avg_color.tolist()
    cap.release()
    return tracks_data

def compute_homography_auto(src_path, dst_path):
    cap1 = cv2.VideoCapture(src_path)
    cap2 = cv2.VideoCapture(dst_path)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cap1.release()
    cap2.release()
    if not ret1 or not ret2:
        raise ValueError("Error reading frames from videos.")

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def load_osnet_model(device):
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    model.eval()
    model.to(device)
    return model

def get_player_center(box):
    return np.array([(box[1] + box[3]) / 2, (box[2] + box[4]) / 2])

def get_average_player_center(boxes, num_frames=5):
    centers = [get_player_center(box) for box in boxes[:num_frames]]
    return np.mean(centers, axis=0) if centers else np.array([0.0, 0.0])

def tune_appearance_weight(tacticam_data, broadcast_data, H, thresholds=np.arange(0.1, 1.0, 0.1)):
    best_weight = 0.5
    best_score = float('inf')
    tac_ids = [tid for tid, data in tacticam_data.items() if 'avg_feature' in data]
    brd_ids = [bid for bid, data in broadcast_data.items() if 'avg_feature' in data]
    if not tac_ids or not brd_ids:
        return best_weight
    tac_feats = np.array([tacticam_data[tid]['avg_feature'] for tid in tac_ids])
    brd_feats = np.array([broadcast_data[bid]['avg_feature'] for bid in brd_ids])
    appearance_cost = cdist(tac_feats, brd_feats, 'cosine')
    spatial_cost = np.full(appearance_cost.shape, np.inf)
    for i, tac_id in enumerate(tac_ids):
        tac_avg_center = get_average_player_center(tacticam_data[tac_id]['boxes']).reshape(1, 1, 2)
        transformed_center = cv2.perspectiveTransform(tac_avg_center, H)[0][0]
        for j, brd_id in enumerate(brd_ids):
            brd_avg_center = get_average_player_center(broadcast_data[brd_id]['boxes'])
            dist = np.linalg.norm(transformed_center - brd_avg_center)
            spatial_cost[i, j] = dist
    spatial_cost_normalized = np.minimum(spatial_cost / SPATIAL_DISTANCE_THRESHOLD, 1.0)
    for w in thresholds:
        combined = (w * appearance_cost) + ((1 - w) * spatial_cost_normalized)
        row_ind, col_ind = linear_sum_assignment(combined)
        avg_cost = np.mean([combined[r, c] for r, c in zip(row_ind, col_ind)])
        if avg_cost < best_score:
            best_score = avg_cost
            best_weight = w
    print(f"\n🔍 Best Appearance Weight Found: {best_weight:.2f} with Avg Match Cost: {best_score:.4f}")
    return best_weight

def match_players(tacticam_data, broadcast_data, H, appearance_weight):
    print("Building cost matrix and matching players...")
    tac_ids = [tid for tid, data in tacticam_data.items() if 'avg_feature' in data]
    brd_ids = [bid for bid, data in broadcast_data.items() if 'avg_feature' in data]
    if not tac_ids or not brd_ids:
        print("⚠️ Warning: Could not find features for matching in one or both videos.")
        return {}
    tac_feats = np.array([tacticam_data[tid]['avg_feature'] for tid in tac_ids])
    brd_feats = np.array([broadcast_data[bid]['avg_feature'] for bid in brd_ids])
    appearance_cost = cdist(tac_feats, brd_feats, 'cosine')
    spatial_cost = np.full(appearance_cost.shape, np.inf)
    for i, tac_id in enumerate(tac_ids):
        tac_avg_center = get_average_player_center(tacticam_data[tac_id]['boxes']).reshape(1, 1, 2)
        transformed_center = cv2.perspectiveTransform(tac_avg_center, H)[0][0]
        for j, brd_id in enumerate(brd_ids):
            brd_avg_center = get_average_player_center(broadcast_data[brd_id]['boxes'])
            dist = np.linalg.norm(transformed_center - brd_avg_center)
            spatial_cost[i, j] = dist
    spatial_cost_normalized = np.minimum(spatial_cost / SPATIAL_DISTANCE_THRESHOLD, 1.0)
    combined_cost = (appearance_weight * appearance_cost) + ((1 - appearance_weight) * spatial_cost_normalized)
    row_ind, col_ind = linear_sum_assignment(combined_cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if combined_cost[r, c] < MATCH_COST_THRESHOLD:
            mapping[tac_ids[r]] = {
                'broadcast_id': brd_ids[c],
                'confidence': 1.0 - combined_cost[r, c]
            }
    return mapping

# --- 3. MAIN EXECUTION PIPELINE ---

def main():
    print(f"Using device: {DEVICE}")
    print("\n--- Step 1: Automatic Homography Estimation ---")
    homography_matrix = compute_homography_auto(TACTICAM_VIDEO, BROADCAST_VIDEO)

    print("\nLoading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    reid_model = load_osnet_model(DEVICE) if USE_OSNET else torch.hub.load(REID_WEIGHTS, REID_MODEL_NAME)
    if not USE_OSNET:
        reid_model.fc = torch.nn.Identity()
    reid_model.to(DEVICE)
    reid_model.eval()

    broadcast_tracks = track_players_in_video(BROADCAST_VIDEO, yolo_model)
    tacticam_tracks = track_players_in_video(TACTICAM_VIDEO, yolo_model)

    broadcast_data = extract_reid_features(BROADCAST_VIDEO, broadcast_tracks, reid_model)
    tacticam_data = extract_reid_features(TACTICAM_VIDEO, tacticam_tracks, reid_model)

    appearance_weight = tune_appearance_weight(tacticam_data, broadcast_data, homography_matrix)
    final_mapping = match_players(tacticam_data, broadcast_data, homography_matrix, appearance_weight)

    print("Saving all output files...")
    with open(os.path.join(OUTPUT_DIR, "final_mapping.json"), 'w') as f:
        json.dump(final_mapping, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "broadcast_features.json"), 'w') as f:
        json.dump(broadcast_data, f)
    with open(os.path.join(OUTPUT_DIR, "tacticam_features.json"), 'w') as f:
        json.dump(tacticam_data, f)

    print("\n✅ Pipeline complete!")
    print("\nFinal Player Mapping (Tacticam ID -> Broadcast ID):")
    print(json.dumps(final_mapping, indent=4))
    print(f"\nAll results saved in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
