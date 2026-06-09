"""
Match BEVFormer camera-only predictions to filtered nuScenes instances.

BEVFormer predicts per-frame 3D boxes (with velocity) but no instance tokens,
so this script matches each target instance's GT box to the closest predicted
car box in the same sample.

Run inside the detr3d (or bevformer) conda environment:

    conda activate detr3d
    python scripts/match_bevformer_instances.py \
        --filtered-csv datasets/filtered_mini.csv \
        --bevformer-results detr3d/work_dirs/bevformer_mini_results.pkl \
        --infos detr3d/data/nuscenes/nuscenes_infos_val.pkl \
        --dataroot datasets/raw \
        --version v1.0-mini

For the full trainval dataset (requires camera blobs downloaded):

    python scripts/match_bevformer_instances.py \
        --filtered-csv datasets/filtered_full.csv \
        --bevformer-results detr3d/work_dirs/bevformer_val_results.pkl \
        --infos detr3d/data/nuscenes/nuscenes_infos_val_full.pkl \
        --dataroot datasets/raw_full \
        --version v1.0-trainval \
        --output outputs/results/bevformer_instance_boxes_full.json
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CAR_LABEL = NUSCENES_CLASSES.index("car")


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def load_results_by_token(results_path, infos_path):
    infos = load_infos(infos_path)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    if len(results) != len(infos):
        raise ValueError(
            f"Result count ({len(results)}) != info count ({len(infos)}). "
            "Ensure the same infos pkl was used for BEVFormer inference."
        )
    return {info["token"]: result for info, result in zip(infos, results)}


def load_filtered_instances(csv_path, val_scenes):
    """Load filtered instances, restricting to val scenes when val_scenes is non-empty."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if val_scenes:
        rows = [r for r in rows if r["scene"] in val_scenes]
    return rows


def iter_scene_samples(nusc, scene_name):
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    token = scene["first_sample_token"]
    frame_idx = 0
    while token:
        sample = nusc.get("sample", token)
        yield frame_idx, sample
        token = sample["next"]
        frame_idx += 1


def find_instance_ann_token(nusc, sample, instance_token):
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["instance_token"] == instance_token:
            return ann_token
    return None


def gt_box_in_lidar(nusc, sample, ann_token):
    lidar_token = sample["data"]["LIDAR_TOP"]
    _, boxes, _ = nusc.get_sample_data(lidar_token, selected_anntokens=[ann_token])
    if not boxes:
        return None
    box = boxes[0]
    return {
        "center_lidar": [float(x) for x in box.center],
        "wlh": [float(x) for x in box.wlh],
        "yaw_lidar": float(box.orientation.yaw_pitch_roll[0]),
    }


def match_closest_car(result, gt_center, score_thr):
    pred = result["pts_bbox"]
    boxes = pred["boxes_3d"].tensor.detach().cpu().numpy()
    scores = pred["scores_3d"].detach().cpu().numpy()
    labels = pred["labels_3d"].detach().cpu().numpy()

    keep = (labels == CAR_LABEL) & (scores >= score_thr)
    if not np.any(keep):
        return None

    kept = np.flatnonzero(keep)
    centers = boxes[kept, :3]
    dists = np.linalg.norm(centers - np.asarray(gt_center)[None, :], axis=1)
    best_local = int(np.argmin(dists))
    best_idx = int(kept[best_local])
    box = boxes[best_idx]

    return {
        "box": {
            # BEVFormer box tensor: [x, y, z, l, w, h, yaw, vx, vy]
            # dims_lwh maps to [length, width, height] — same convention as GT metadata
            "center_lidar": [float(x) for x in box[:3]],
            "dims_lwh": [float(x) for x in box[3:6]],
            "yaw_lidar": float(box[6]),
            "velocity_lidar": (
                [float(box[7]), float(box[8])] if box.shape[0] >= 9 else [0.0, 0.0]
            ),
        },
        "score": float(scores[best_idx]),
        "label": NUSCENES_CLASSES[int(labels[best_idx])],
        "center_distance_m": float(dists[best_local]),
        "prediction_index": int(best_idx),
    }


def count_complete_windows(matches_by_instance, T):
    """Report how many T-consecutive windows have all frames matched (for coverage info)."""
    total_windows = 0
    complete_windows = 0
    for frames in matches_by_instance.values():
        frames_sorted = sorted(frames, key=lambda r: r["frame"])
        matched_set = {r["frame"] for r in frames_sorted if r["matched"] is not None}
        for i in range(len(frames_sorted) - T + 1):
            window = frames_sorted[i:i + T]
            if any(window[j + 1]["frame"] != window[j]["frame"] + 1 for j in range(T - 1)):
                continue
            total_windows += 1
            if all(w["frame"] in matched_set for w in window):
                complete_windows += 1
    return total_windows, complete_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-csv", default="datasets/filtered_mini.csv")
    parser.add_argument("--bevformer-results",
                        default="detr3d/work_dirs/bevformer_mini_results.pkl")
    parser.add_argument("--infos",
                        default="detr3d/data/nuscenes/nuscenes_infos_val.pkl")
    parser.add_argument("--dataroot", default="datasets/raw")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--score-thr", type=float, default=0.3,
                        help="Minimum BEVFormer confidence score to consider a prediction.")
    parser.add_argument("--max-center-distance", type=float, default=2.0,
                        help="Reject match if predicted center is farther than this (metres).")
    parser.add_argument("--T", type=int, default=4,
                        help="Temporal window size (for coverage reporting only).")
    parser.add_argument("--no-val-filter", action="store_true",
                        help="Skip val-scene filtering (use all scenes in the CSV).")
    parser.add_argument("--output",
                        default="outputs/results/bevformer_instance_boxes.json")
    args = parser.parse_args()

    if args.no_val_filter:
        val_scenes = set()
    else:
        splits = create_splits_scenes()
        val_scenes = set(splits["val"])

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    filtered = load_filtered_instances(args.filtered_csv, val_scenes)
    print(f"Processing {len(filtered)} instances "
          f"({'all scenes' if args.no_val_filter else 'val scenes only'})")

    results_by_token = load_results_by_token(args.bevformer_results, args.infos)

    matches = []
    matches_by_instance = {}
    summary = {
        "targets": len(filtered),
        "frames_with_gt_instance": 0,
        "frames_with_bevformer_prediction": 0,
        "matched_frames": 0,
        "unmatched_frames": 0,
        "missing_prediction_frames": 0,
    }

    for target in filtered:
        scene_name = target["scene"]
        instance_token = target["instance_token"]
        inst_key = (scene_name, instance_token)

        for frame_idx, sample in iter_scene_samples(nusc, scene_name):
            ann_token = find_instance_ann_token(nusc, sample, instance_token)
            if ann_token is None:
                continue

            summary["frames_with_gt_instance"] += 1
            gt_box = gt_box_in_lidar(nusc, sample, ann_token)
            if gt_box is None:
                continue

            result = results_by_token.get(sample["token"])
            if result is None:
                summary["missing_prediction_frames"] += 1
                rec = {
                    "scene": scene_name, "frame": frame_idx,
                    "sample_token": sample["token"],
                    "ann_token": ann_token,
                    "instance_token": instance_token,
                    "gt_box": gt_box, "matched": None,
                    "status": "missing_bevformer_prediction_for_sample",
                }
                matches.append(rec)
                matches_by_instance.setdefault(inst_key, []).append(rec)
                continue

            summary["frames_with_bevformer_prediction"] += 1
            matched = match_closest_car(result, gt_box["center_lidar"], args.score_thr)

            if matched is None:
                status = "no_car_prediction_above_score_threshold"
                summary["unmatched_frames"] += 1
            elif matched["center_distance_m"] > args.max_center_distance:
                matched = None
                status = "closest_car_prediction_too_far"
                summary["unmatched_frames"] += 1
            else:
                status = "matched"
                summary["matched_frames"] += 1

            rec = {
                "scene": scene_name, "frame": frame_idx,
                "sample_token": sample["token"],
                "ann_token": ann_token,
                "instance_token": instance_token,
                "gt_box": gt_box, "matched": matched,
                "status": status,
            }
            matches.append(rec)
            matches_by_instance.setdefault(inst_key, []).append(rec)

    total_windows, complete_windows = count_complete_windows(matches_by_instance, args.T)
    summary["T"] = args.T
    summary["total_consecutive_windows"] = total_windows
    summary["complete_windows_all_matched"] = complete_windows

    output = {
        "score_threshold": args.score_thr,
        "max_center_distance": args.max_center_distance,
        "class_filter": "car",
        "summary": summary,
        "matches": matches,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved {len(matches)} frame records → {out_path}")


if __name__ == "__main__":
    main()
