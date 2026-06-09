"""
Match DETR3D camera-only predictions to selected nuScenes instances.

The filtered CSV contains nuScenes instance tokens from ground-truth tracking.
DETR3D predicts per-frame 3D boxes but does not output instance tokens, so this
script matches each target instance's GT box to the closest predicted car box
in the same sample.

Run with the DETR3D conda environment because the result pickle contains
MMDetection3D box objects:

    conda activate detr3d
    python scripts/match_detr3d_instances.py \
        --filtered-csv datasets/filtered_mini.csv \
        --detr3d-results detr3d/work_dirs/detr3d_mini_results.pkl \
        --infos detr3d/data/nuscenes/nuscenes_infos_val.pkl \
        --dataroot datasets/raw \
        --version v1.0-mini
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes


NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
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
            f"Result count ({len(results)}) does not match info count ({len(infos)}). "
            "Use the same infos pkl that DETR3D inference used."
        )

    return {info["token"]: result for info, result in zip(infos, results)}


def load_filtered_instances(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


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
    yaw = float(box.orientation.yaw_pitch_roll[0])
    return {
        "center_lidar": [float(x) for x in box.center],
        "wlh": [float(x) for x in box.wlh],
        "yaw_lidar": yaw,
    }


def prediction_arrays(result):
    pred = result["pts_bbox"]
    boxes = pred["boxes_3d"].tensor.detach().cpu().numpy()
    scores = pred["scores_3d"].detach().cpu().numpy()
    labels = pred["labels_3d"].detach().cpu().numpy()
    return boxes, scores, labels


def match_closest_car(result, gt_center, score_thr):
    boxes, scores, labels = prediction_arrays(result)
    keep = (labels == CAR_LABEL) & (scores >= score_thr)
    if not np.any(keep):
        return None

    kept_indices = np.flatnonzero(keep)
    centers = boxes[kept_indices, :3]
    dists = np.linalg.norm(centers - np.asarray(gt_center)[None, :], axis=1)
    best_local = int(np.argmin(dists))
    best_idx = int(kept_indices[best_local])
    box = boxes[best_idx]

    return {
        "box": {
            "center_lidar": [float(x) for x in box[:3]],
            "dims_lwh_or_dxdydz": [float(x) for x in box[3:6]],
            "yaw_lidar": float(box[6]),
            "velocity_lidar": [float(x) for x in box[7:9]] if box.shape[0] >= 9 else None,
        },
        "score": float(scores[best_idx]),
        "label": NUSCENES_CLASSES[int(labels[best_idx])],
        "center_distance_m": float(dists[best_local]),
        "prediction_index": best_idx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-csv", default="datasets/filtered_mini.csv")
    parser.add_argument("--detr3d-results", default="detr3d/work_dirs/detr3d_mini_results.pkl")
    parser.add_argument("--infos", default="detr3d/data/nuscenes/nuscenes_infos_val.pkl")
    parser.add_argument("--dataroot", default="datasets/raw")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=1.0,
        help="Reject closest car prediction if its center is farther than this many metres.",
    )
    parser.add_argument(
        "--output",
        default="outputs/results/detr3d_instance_boxes_mini.json",
    )
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    filtered = load_filtered_instances(args.filtered_csv)
    results_by_token = load_results_by_token(args.detr3d_results, args.infos)

    matches = []
    summary = {
        "targets": len(filtered),
        "frames_with_gt_instance": 0,
        "frames_with_detr3d_prediction": 0,
        "matched_frames": 0,
        "unmatched_frames": 0,
        "missing_prediction_frames": 0,
    }

    for target in filtered:
        scene_name = target["scene"]
        instance_token = target["instance_token"]

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
                matches.append({
                    "scene": scene_name,
                    "frame": frame_idx,
                    "sample_token": sample["token"],
                    "ann_token": ann_token,
                    "instance_token": instance_token,
                    "gt_box": gt_box,
                    "matched": None,
                    "status": "missing_detr3d_prediction_for_sample",
                })
                continue

            summary["frames_with_detr3d_prediction"] += 1
            matched = match_closest_car(result, gt_box["center_lidar"], args.score_thr)
            if matched is None:
                summary["unmatched_frames"] += 1
                status = "no_car_prediction_above_score_threshold"
            elif matched["center_distance_m"] > args.max_center_distance:
                matched = None
                summary["unmatched_frames"] += 1
                status = "closest_car_prediction_too_far"
            else:
                summary["matched_frames"] += 1
                status = "matched"

            matches.append({
                "scene": scene_name,
                "frame": frame_idx,
                "sample_token": sample["token"],
                "ann_token": ann_token,
                "instance_token": instance_token,
                "gt_box": gt_box,
                "matched": matched,
                "status": status,
            })

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
    print(f"Saved {len(matches)} frame records to {out_path}")


if __name__ == "__main__":
    main()
