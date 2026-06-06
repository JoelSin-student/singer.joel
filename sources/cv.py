import argparse
import csv
import json
import math
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sources.loader import load_config
from sources.predict import get_parser as get_predict_parser
from sources.predict import start as start_predict
from sources.train import get_parser as get_train_parser
from sources.train import start as start_train
from sources.util import join_nonempty, prepare_runtime_data_from_pre_split_cache


@dataclass
class SampleKey:
    key: str
    subject: str
    situation_id: str
    situation_name: str


def _safe_slug(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def _short_hash(text, length=6):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def _fold_code(fold):
    subject = _safe_slug(fold.get("held_out_subject", ""))
    situation = _safe_slug(fold.get("held_out_situation", ""))
    fold_id = _safe_slug(fold.get("fold_id", "fold"))

    if fold.get("held_out_subject") and fold.get("held_out_situation"):
        base = f"{subject}_{situation}"
    elif fold.get("held_out_subject"):
        base = f"sub_{subject}"
    elif fold.get("held_out_situation"):
        base = f"sit_{situation}"
    else:
        base = fold_id

    return f"{base}_{_short_hash(fold_id)}"


def _parse_sample_key(key):
    key_str = str(key).strip()
    parts = key_str.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected key format: '{key_str}'")

    subject = parts[0]
    situation_id = parts[1]
    situation_name = "_".join(parts[1:])
    return SampleKey(
        key=key_str,
        subject=subject,
        situation_id=situation_id,
        situation_name=situation_name,
    )


def _extract_subject_key_from_tag(tag):
    parts = str(tag).strip().split("_")
    return parts[0] if parts else ""


def _load_subject_height_map(root):
    subject_info_path = Path(root) / "data" / "subject_info.txt"
    if not subject_info_path.is_file():
        raise FileNotFoundError(f"subject_info.txt not found: {subject_info_path}")

    rows = np.genfromtxt(
        subject_info_path,
        delimiter=None,
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    dtype_names = list(rows.dtype.names or [])
    if "subject_key" not in dtype_names or "height" not in dtype_names:
        raise ValueError(
            "subject_info.txt must include 'subject_key' and 'height' columns for denormalization."
        )

    if rows.ndim == 0:
        rows = np.asarray([rows])

    height_map = {}
    for row in rows:
        key = str(row["subject_key"]).strip()
        if not key:
            continue
        height_map[key] = float(row["height"])
    return height_map


def _looks_height_normalized(pos, max_abs_threshold=0.02):
    stacked = np.stack([pos["X"], pos["Y"], pos["Z"]], axis=0)
    finite = np.isfinite(stacked)
    if not np.any(finite):
        return False
    max_abs = float(np.nanmax(np.abs(stacked[finite])))
    return max_abs <= float(max_abs_threshold)


def _denormalize_by_subject_height_if_needed(pos, subject_height):
    if not _looks_height_normalized(pos):
        return pos, False

    scale = float(subject_height)
    return (
        {
            "X": pos["X"] * scale,
            "Y": pos["Y"] * scale,
            "Z": pos["Z"] * scale,
            "labels": list(pos["labels"]),
        },
        True,
    )


def _sorted_ids(values):
    def _key(v):
        token = str(v)
        return (0, int(token)) if token.isdigit() else (1, token)

    return sorted(set(values), key=_key)


def _discover_available_keys(clean_data_dir):
    clean_data_dir = Path(clean_data_dir)
    if not clean_data_dir.is_dir():
        raise FileNotFoundError(f"clean_data folder not found: {clean_data_dir}")

    skeleton_keys = {
        path.stem.split("_", 1)[1]
        for path in clean_data_dir.glob("Awinda_*.csv")
        if path.is_file() and "_" in path.stem
    }
    insole_keys = {
        path.stem.split("_", 1)[1]
        for path in clean_data_dir.glob("Soles_*.txt")
        if path.is_file() and "_" in path.stem
    }
    common = sorted(skeleton_keys & insole_keys)
    if not common:
        raise ValueError(
            "No common Awinda/Soles keys found in clean_data. "
            "Make sure preprocessing generated both file types."
        )

    entries = [_parse_sample_key(key) for key in common]
    return entries


def _extract_numeric_suffix(text):
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else math.inf


def _sort_situations(values):
    return sorted(set(values), key=lambda s: (_extract_numeric_suffix(s), str(s)))


def _build_loso_folds(entries):
    folds = []
    for subject in _sorted_ids([e.subject for e in entries]):
        test_keys = [e.key for e in entries if e.subject == subject]
        train_keys = [e.key for e in entries if e.subject != subject]
        folds.append(
            {
                "fold_id": f"loso_{subject}",
                "held_out_subject": subject,
                "held_out_situation": "",
                "train_keys": train_keys,
                "test_keys": test_keys,
            }
        )
    return folds


def _build_losito_folds(entries):
    folds = []
    for situation in _sort_situations([e.situation_name for e in entries]):
        test_keys = [e.key for e in entries if e.situation_name == situation]
        train_keys = [e.key for e in entries if e.situation_name != situation]
        folds.append(
            {
                "fold_id": f"losito_{situation}",
                "held_out_subject": "",
                "held_out_situation": situation,
                "train_keys": train_keys,
                "test_keys": test_keys,
            }
        )
    return folds


def _build_ps_losito_fold_groups(entries):
    groups = {}
    for subject in _sorted_ids([e.subject for e in entries]):
        subject_entries = [e for e in entries if e.subject == subject]
        folds = []
        for situation in _sort_situations([e.situation_name for e in subject_entries]):
            test_keys = [e.key for e in subject_entries if e.situation_name == situation]
            train_keys = [e.key for e in subject_entries if e.situation_name != situation]
            folds.append(
                {
                    "fold_id": f"ps_losito_{subject}_{situation}",
                    "held_out_subject": subject,
                    "held_out_situation": situation,
                    "train_keys": train_keys,
                    "test_keys": test_keys,
                }
            )
        groups[subject] = folds
    return groups


def _build_psit_loso_fold_groups(entries):
    groups = {}
    for situation in _sort_situations([e.situation_name for e in entries]):
        situation_entries = [e for e in entries if e.situation_name == situation]
        folds = []
        for subject in _sorted_ids([e.subject for e in situation_entries]):
            test_keys = [
                e.key
                for e in situation_entries
                if e.subject == subject
            ]
            train_keys = [
                e.key
                for e in situation_entries
                if e.subject != subject
            ]
            folds.append(
                {
                    "fold_id": f"psit_loso_{situation}_{subject}",
                    "held_out_subject": subject,
                    "held_out_situation": situation,
                    "train_keys": train_keys,
                    "test_keys": test_keys,
                }
            )
        groups[situation] = folds
    return groups


def _align_on_frame(gt_df, pred_df):
    if "Frame" in gt_df.columns and "Frame" in pred_df.columns:
        merged = pd.merge(gt_df, pred_df, on="Frame", suffixes=("_gt", "_pred"))
        gt_aligned = merged[[c for c in merged.columns if c.endswith("_gt")]].copy()
        pred_aligned = merged[[c for c in merged.columns if c.endswith("_pred")]].copy()
        gt_aligned.columns = [c[:-3] for c in gt_aligned.columns]
        pred_aligned.columns = [c[:-5] for c in pred_aligned.columns]
        return gt_aligned, pred_aligned

    n = min(len(gt_df), len(pred_df))
    return gt_df.iloc[:n].copy(), pred_df.iloc[:n].copy()


def _extract_xyz(df):
    cols = list(df.columns)

    x_cols = [c for c in cols if re.match(r"^X\.\d+$", str(c))]
    y_cols = [c for c in cols if re.match(r"^Y\.\d+$", str(c))]
    z_cols = [c for c in cols if re.match(r"^Z\.\d+$", str(c))]

    if x_cols and y_cols and z_cols:
        x_cols = sorted(x_cols, key=lambda c: int(str(c).split(".")[1]))
        y_cols = sorted(y_cols, key=lambda c: int(str(c).split(".")[1]))
        z_cols = sorted(z_cols, key=lambda c: int(str(c).split(".")[1]))
        labels = [str(i) for i in range(len(x_cols))]
        return {
            "X": df[x_cols].to_numpy(dtype=np.float64),
            "Y": df[y_cols].to_numpy(dtype=np.float64),
            "Z": df[z_cols].to_numpy(dtype=np.float64),
            "labels": labels,
        }

    pos_cols = [c for c in cols if str(c).startswith("pos::")]
    if not pos_cols:
        raise ValueError(
            "Could not extract XYZ columns. Supported formats: X.N/Y.N/Z.N or pos::... x/y/z"
        )

    x_map = {}
    y_map = {}
    z_map = {}
    for col in pos_cols:
        col_str = str(col)
        if col_str.endswith(" x"):
            x_map[col_str[:-2]] = col
        elif col_str.endswith(" y"):
            y_map[col_str[:-2]] = col
        elif col_str.endswith(" z"):
            z_map[col_str[:-2]] = col

    labels = [label for label in x_map.keys() if label in y_map and label in z_map]
    if not labels:
        raise ValueError("No common pos:: labels with x/y/z axes were found.")

    return {
        "X": df[[x_map[label] for label in labels]].to_numpy(dtype=np.float64),
        "Y": df[[y_map[label] for label in labels]].to_numpy(dtype=np.float64),
        "Z": df[[z_map[label] for label in labels]].to_numpy(dtype=np.float64),
        "labels": labels,
    }


def _align_joint_sets(gt_pos, pred_pos):
    gt_labels = [str(x) for x in gt_pos["labels"]]
    pred_labels = [str(x) for x in pred_pos["labels"]]
    common = [label for label in gt_labels if label in set(pred_labels)]
    if not common:
        raise ValueError("No common joint labels between GT and prediction.")

    gt_idx = [gt_labels.index(label) for label in common]
    pred_idx = [pred_labels.index(label) for label in common]

    return (
        {
            "X": gt_pos["X"][:, gt_idx],
            "Y": gt_pos["Y"][:, gt_idx],
            "Z": gt_pos["Z"][:, gt_idx],
            "labels": common,
        },
        {
            "X": pred_pos["X"][:, pred_idx],
            "Y": pred_pos["Y"][:, pred_idx],
            "Z": pred_pos["Z"][:, pred_idx],
            "labels": common,
        },
    )


def _exclude_joint_index_zero(pos):
    # Metrics are reported without the root/pelvis joint at index 0.
    if pos["X"].shape[1] <= 1:
        raise ValueError("Cannot exclude joint index 0 because fewer than 2 joints are available.")

    return {
        "X": pos["X"][:, 1:],
        "Y": pos["Y"][:, 1:],
        "Z": pos["Z"][:, 1:],
        "labels": list(pos["labels"])[1:],
    }


def _compute_velocity(pos, fps=60.0):
    if pos["X"].shape[0] < 2:
        return {
            "X": np.empty((0, pos["X"].shape[1]), dtype=np.float64),
            "Y": np.empty((0, pos["Y"].shape[1]), dtype=np.float64),
            "Z": np.empty((0, pos["Z"].shape[1]), dtype=np.float64),
            "labels": pos["labels"],
        }

    return {
        "X": (pos["X"][1:, :] - pos["X"][:-1, :]) * fps,
        "Y": (pos["Y"][1:, :] - pos["Y"][:-1, :]) * fps,
        "Z": (pos["Z"][1:, :] - pos["Z"][:-1, :]) * fps,
        "labels": pos["labels"],
    }


def _compute_mpjpe(gt, pred):
    dx = pred["X"] - gt["X"]
    dy = pred["Y"] - gt["Y"]
    dz = pred["Z"] - gt["Z"]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return {
        "full": float(np.nanmean(dist)),
        "per_joint": np.nanmean(dist, axis=0),
        "matrix": dist,
    }


def _compute_pa_mpjpe(gt, pred):
    n_frames, n_joints = gt["X"].shape
    dist = np.full((n_frames, n_joints), np.nan, dtype=np.float64)

    for i in range(n_frames):
        g = np.column_stack([gt["X"][i, :], gt["Y"][i, :], gt["Z"][i, :]])
        p = np.column_stack([pred["X"][i, :], pred["Y"][i, :], pred["Z"][i, :]])

        gc = g - np.mean(g, axis=0, keepdims=True)
        pc = p - np.mean(p, axis=0, keepdims=True)

        h = pc.T @ gc
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1.0
            r = vt.T @ u.T

        p_aligned = pc @ r
        dist[i, :] = np.sqrt(np.sum((p_aligned - gc) ** 2, axis=1))

    return {
        "full": float(np.nanmean(dist)),
        "per_joint": np.nanmean(dist, axis=0),
        "matrix": dist,
    }


def _compute_inconsistency(gt, pred):
    dx = pred["X"] - gt["X"]
    dy = pred["Y"] - gt["Y"]
    dz = pred["Z"] - gt["Z"]

    per_joint = np.nanmean(
        np.stack(
            [
                np.nanstd(dx, axis=0),
                np.nanstd(dy, axis=0),
                np.nanstd(dz, axis=0),
            ],
            axis=0,
        ),
        axis=0,
    )
    return {
        "full": float(np.nanmean(per_joint)),
        "per_joint": per_joint,
    }


def _compute_mpjve(gt, pred, fps=60.0):
    gt_v = _compute_velocity(gt, fps=fps)
    pred_v = _compute_velocity(pred, fps=fps)
    return _compute_mpjpe(gt_v, pred_v)


def _compute_acceleration(pos, fps=60.0):
    if pos["X"].shape[0] < 3:
        return {
            "X": np.empty((0, pos["X"].shape[1]), dtype=np.float64),
            "Y": np.empty((0, pos["Y"].shape[1]), dtype=np.float64),
            "Z": np.empty((0, pos["Z"].shape[1]), dtype=np.float64),
            "labels": pos["labels"],
        }

    scale = float(fps) ** 2
    return {
        "X": (pos["X"][2:, :] - 2.0 * pos["X"][1:-1, :] + pos["X"][:-2, :]) * scale,
        "Y": (pos["Y"][2:, :] - 2.0 * pos["Y"][1:-1, :] + pos["Y"][:-2, :]) * scale,
        "Z": (pos["Z"][2:, :] - 2.0 * pos["Z"][1:-1, :] + pos["Z"][:-2, :]) * scale,
        "labels": pos["labels"],
    }


def _compute_mpjace(gt, pred, fps=60.0):
    gt_a = _compute_acceleration(gt, fps=fps)
    pred_a = _compute_acceleration(pred, fps=fps)
    if gt_a["X"].shape[0] == 0 or pred_a["X"].shape[0] == 0:
        empty = np.full((len(gt["labels"]),), np.nan, dtype=np.float64)
        return {"full": np.nan, "per_joint": empty}

    dx = pred_a["X"] - gt_a["X"]
    dy = pred_a["Y"] - gt_a["Y"]
    dz = pred_a["Z"] - gt_a["Z"]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return {
        "full": float(np.nanmean(dist)),
        "per_joint": np.nanmean(dist, axis=0),
    }


def _compute_motion_energy(pos, fps=60.0):
    v = _compute_velocity(pos, fps=fps)
    energy_per_joint = np.nanmean(v["X"] ** 2 + v["Y"] ** 2 + v["Z"] ** 2, axis=0)
    return {
        "per_joint": energy_per_joint,
        "full": float(np.nanmean(energy_per_joint)),
    }


def _compute_mer(gt, pred, fps=60.0):
    e_gt = _compute_motion_energy(gt, fps=fps)
    e_pred = _compute_motion_energy(pred, fps=fps)

    ratio_per_joint = e_pred["per_joint"] / e_gt["per_joint"]
    ratio_per_joint[~np.isfinite(ratio_per_joint)] = np.nan
    ratio_full = e_pred["full"] / e_gt["full"] if e_gt["full"] != 0 else np.nan

    return {
        "full": float(ratio_full),
        "per_joint": ratio_per_joint,
    }


def _json_vector(vec):
    return json.dumps([None if not np.isfinite(float(v)) else float(v) for v in vec])


def _evaluate_fold_predictions(root, prediction_rows, fps=60.0):
    root = Path(root)
    all_gt_x = []
    all_gt_y = []
    all_gt_z = []
    all_pr_x = []
    all_pr_y = []
    all_pr_z = []
    labels = None
    pair_count = 0
    height_map = _load_subject_height_map(root)
    denorm_gt_count = 0
    denorm_pred_count = 0

    for row in prediction_rows:
        tag = str(row["tag"]).strip()
        pred_file = Path(row["output_file"])
        gt_file = root / "data" / "test_data" / "skeleton" / f"Awinda_{tag}.csv"

        if not pred_file.is_file():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        if not gt_file.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {gt_file}")

        gt_df = pd.read_csv(gt_file)
        pred_df = pd.read_csv(pred_file)
        gt_df, pred_df = _align_on_frame(gt_df, pred_df)

        gt_pos = _extract_xyz(gt_df)
        pred_pos = _extract_xyz(pred_df)
        gt_pos, pred_pos = _align_joint_sets(gt_pos, pred_pos)

        subject_key = _extract_subject_key_from_tag(tag)
        if subject_key not in height_map:
            raise KeyError(
                f"Missing subject height for '{subject_key}' in data/subject_info.txt"
            )
        subject_height = float(height_map[subject_key])

        gt_pos, gt_denorm_applied = _denormalize_by_subject_height_if_needed(gt_pos, subject_height)
        pred_pos, pred_denorm_applied = _denormalize_by_subject_height_if_needed(pred_pos, subject_height)
        denorm_gt_count += int(gt_denorm_applied)
        denorm_pred_count += int(pred_denorm_applied)

        gt_pos = _exclude_joint_index_zero(gt_pos)
        pred_pos = _exclude_joint_index_zero(pred_pos)

        if labels is None:
            labels = list(gt_pos["labels"])

        all_gt_x.append(gt_pos["X"])
        all_gt_y.append(gt_pos["Y"])
        all_gt_z.append(gt_pos["Z"])
        all_pr_x.append(pred_pos["X"])
        all_pr_y.append(pred_pos["Y"])
        all_pr_z.append(pred_pos["Z"])
        pair_count += 1

    if not all_gt_x:
        raise ValueError("No prediction/GT pairs were found for evaluation.")

    print(
        "[CV] Height denormalization summary: "
        f"gt_applied={denorm_gt_count}/{pair_count}, "
        f"pred_applied={denorm_pred_count}/{pair_count}"
    )

    gt = {
        "X": np.vstack(all_gt_x),
        "Y": np.vstack(all_gt_y),
        "Z": np.vstack(all_gt_z),
        "labels": labels,
    }
    pred = {
        "X": np.vstack(all_pr_x),
        "Y": np.vstack(all_pr_y),
        "Z": np.vstack(all_pr_z),
        "labels": labels,
    }

    mpjpe = _compute_mpjpe(gt, pred)
    pa = _compute_pa_mpjpe(gt, pred)
    inconsistency = _compute_inconsistency(gt, pred)
    mpjve = _compute_mpjve(gt, pred, fps=fps)
    mpjace = _compute_mpjace(gt, pred, fps=fps)
    mer = _compute_mer(gt, pred, fps=fps)

    total_frames = int(gt["X"].shape[0])
    labels_out = labels if labels is not None else []
    return {
        "num_pairs": int(pair_count),
        "num_frames": total_frames,
        "num_joints": int(gt["X"].shape[1]),
        "joint_labels_json": json.dumps([str(x) for x in labels_out]),
        "mpjpe_full": float(mpjpe["full"]),
        "pa_mpjpe_full": float(pa["full"]),
        "inconsistency_full": float(inconsistency["full"]),
        "mpjve_full": float(mpjve["full"]),
        "mpjace_full": float(mpjace["full"]),
        "mer_full": float(mer["full"]),
        "mpjpe_per_joint_json": _json_vector(mpjpe["per_joint"]),
        "pa_mpjpe_per_joint_json": _json_vector(pa["per_joint"]),
        "inconsistency_per_joint_json": _json_vector(inconsistency["per_joint"]),
        "mpjve_per_joint_json": _json_vector(mpjve["per_joint"]),
        "mpjace_per_joint_json": _json_vector(mpjace["per_joint"]),
        "mer_per_joint_json": _json_vector(mer["per_joint"]),
    }


def _mean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _std_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanstd(arr))


def _append_aggregate_rows(rows):
    metric_keys = [
        "mpjpe_full",
        "pa_mpjpe_full",
        "inconsistency_full",
        "mpjve_full",
        "mpjace_full",
        "mer_full",
        "inference_fps",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    out = list(rows)
    for algorithm, algo_rows in grouped.items():
        mean_row = {
            **{k: "" for k in rows[0].keys()},
            "summary_kind": "aggregate_mean",
            "fold_id": "aggregate_mean",
            "algorithm": algorithm,
            "model_mode": algo_rows[0]["model_mode"],
        }
        std_row = {
            **{k: "" for k in rows[0].keys()},
            "summary_kind": "aggregate_std",
            "fold_id": "aggregate_std",
            "algorithm": algorithm,
            "model_mode": algo_rows[0]["model_mode"],
        }

        for key in metric_keys:
            values = [row[key] for row in algo_rows]
            mean_row[key] = _mean_or_nan(values)
            std_row[key] = _std_or_nan(values)

        out.append(mean_row)
        out.append(std_row)

    return out


def _write_summary_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty summary: {output_path}")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_train_args(config_path, model, model_mode, abl_id):
    parser = get_train_parser(add_help=False)
    args = parser.parse_args([])
    args.mode = "train"
    args.model = model
    args.config = str(config_path)
    args.model_mode = model_mode
    args.cv_enable = "false"
    args.abl_id = str(abl_id)
    return args


def _build_predict_args(config_path, model, model_mode, abl_id, checkpoint_file):
    parser = get_predict_parser(add_help=False)
    args = parser.parse_args([])
    args.mode = "predict"
    args.model = model
    args.config = str(config_path)
    args.model_mode = model_mode
    args.abl_id = str(abl_id)
    args.checkpoint_file = str(checkpoint_file)
    return args


def _best_checkpoint_path(model_mode, abl_id):
    stem = join_nonempty("best_skeleton_model", f"abl_id_{abl_id}", model_mode)
    return Path(".") / "results" / "weight" / f"{stem}.pth"


def _run_fold(
    root,
    fold,
    algorithm,
    model_mode,
    train_config,
    predict_config,
    model,
    fps,
    summary_label,
    run_label,
):
    train_keys = list(fold["train_keys"])
    test_keys = list(fold["test_keys"])
    if not train_keys or not test_keys:
        raise ValueError(
            f"Invalid fold '{fold['fold_id']}': train={len(train_keys)} test={len(test_keys)}"
        )

    prepare_runtime_data_from_pre_split_cache(
        root=Path(root),
        train_files=train_keys,
        test_files=test_keys,
    )

    run_tag = _safe_slug(f"{run_label}_{_fold_code(fold)}")
    train_args = _build_train_args(train_config, model, model_mode, run_tag)
    start_train(train_args)

    checkpoint_file = _best_checkpoint_path(model_mode, run_tag)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_file}")

    predict_args = _build_predict_args(
        predict_config,
        model,
        model_mode,
        run_tag,
        checkpoint_file,
    )
    predict_result = start_predict(predict_args)
    if not isinstance(predict_result, dict) or "outputs" not in predict_result:
        raise RuntimeError("Prediction did not return output metadata for evaluation.")

    eval_result = _evaluate_fold_predictions(
        root=Path(root),
        prediction_rows=predict_result["outputs"],
        fps=fps,
    )

    return {
        "summary_kind": "fold",
        "cv_type": summary_label,
        "fold_id": fold["fold_id"],
        "held_out_subject": fold.get("held_out_subject", ""),
        "held_out_situation": fold.get("held_out_situation", ""),
        "num_train_keys": int(len(train_keys)),
        "num_test_keys": int(len(test_keys)),
        "algorithm": algorithm,
        "model_mode": model_mode,
        "run_tag": run_tag,
        "checkpoint_file": str(checkpoint_file),
        "prediction_cycle_tag": str(predict_result.get("cycle_tag", "")),
        "inference_fps": float(predict_result.get("inference_fps", np.nan)),
        "inference_seconds": float(predict_result.get("inference_seconds", np.nan)),
        "predicted_frames": int(predict_result.get("predicted_frames", 0)),
        **eval_result,
    }


def _build_cv_plan(entries):
    plan = []

    plan.append(
        {
            "cv_family": "loso",
            "group_key": "",
            "summary_label": "LOSO",
            "summary_slug": "loso",
            "folds": _build_loso_folds(entries),
        }
    )
    plan.append(
        {
            "cv_family": "losito",
            "group_key": "",
            "summary_label": "LOSitO",
            "summary_slug": "losito",
            "folds": _build_losito_folds(entries),
        }
    )

    ps_losito = _build_ps_losito_fold_groups(entries)
    for subject in _sorted_ids(ps_losito.keys()):
        plan.append(
            {
                "cv_family": "ps-losito",
                "group_key": str(subject),
                "summary_label": f"PS-LOSitO[{subject}]",
                "summary_slug": f"ps_losito_{_safe_slug(subject)}",
                "folds": ps_losito[subject],
            }
        )

    psit_loso = _build_psit_loso_fold_groups(entries)
    for situation in _sort_situations(psit_loso.keys()):
        plan.append(
            {
                "cv_family": "psit-loso",
                "group_key": str(situation),
                "summary_label": f"PSit-LOSO[{situation}]",
                "summary_slug": f"psit_loso_{_safe_slug(situation)}",
                "folds": psit_loso[situation],
            }
        )

    return plan


def _parse_csv_list(text, fallback):
    value = str(text).strip() if text is not None else ""
    if not value:
        return list(fallback)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items if items else list(fallback)


def _normalize_selector(text):
    return _safe_slug(str(text).strip())


def start(args):
    config = load_config(args, args.config, args.model)
    root = Path(__file__).resolve().parents[1]

    cv_cfg = config.get("cv", {})
    loc_cfg = config.get("location", {})

    clean_data_dir = loc_cfg.get("clean_data_dir", os.path.join("data", "clean_data"))
    if not os.path.isabs(clean_data_dir):
        clean_data_dir = root / clean_data_dir

    output_dir = cv_cfg.get("output_dir", os.path.join("results", "final_eval", "cv"))
    if not os.path.isabs(output_dir):
        output_dir = root / output_dir

    train_config = cv_cfg.get("train_config", os.path.join("sources", "config", "transformer_encoder", "train.yaml"))
    if not os.path.isabs(train_config):
        train_config = root / train_config

    predict_config = cv_cfg.get("predict_config", os.path.join("sources", "config", "transformer_encoder", "predict.yaml"))
    if not os.path.isabs(predict_config):
        predict_config = root / predict_config

    run_label_cfg = cv_cfg.get("run_label", "default")
    run_label = _safe_slug(args.run_label if args.run_label is not None else run_label_cfg)

    algorithm_names = _parse_csv_list(
        args.algorithms,
        cv_cfg.get("algorithms", ["p2p_insole", "soleformer"]),
    )
    algorithm_map = {
        "p2p_insole": "original",
        "p2p": "original",
        "original": "original",
        "soleformer": "soleformer",
    }
    algorithms = []
    for name in algorithm_names:
        key = str(name).strip().lower()
        if key not in algorithm_map:
            raise ValueError(
                f"Unsupported algorithm '{name}'. Supported: {sorted(set(algorithm_map.keys()))}"
            )
        algorithms.append((key, algorithm_map[key]))

    fps = float(args.fps if args.fps is not None else cv_cfg.get("fps", 60.0))

    run_types = _parse_csv_list(
        args.run_types,
        cv_cfg.get("run_types", ["LOSO", "LOSitO", "PS-LOSitO", "PSit-LOSO"]),
    )
    normalized_run_types = {t.strip().lower() for t in run_types}

    selected_ps_losito_subjects = _parse_csv_list(
        args.ps_losito_subjects,
        cv_cfg.get("ps_losito_subjects", []),
    )
    selected_psit_loso_situations = _parse_csv_list(
        args.psit_loso_situations,
        cv_cfg.get("psit_loso_situations", []),
    )

    selected_subject_tokens = {_normalize_selector(s) for s in selected_ps_losito_subjects}
    selected_situation_tokens = {_normalize_selector(s) for s in selected_psit_loso_situations}

    entries = _discover_available_keys(clean_data_dir)
    plan = _build_cv_plan(entries)

    wanted_plans = []
    for item in plan:
        family = item.get("cv_family", "")
        group_key = str(item.get("group_key", ""))

        if family == "loso" and "loso" in normalized_run_types:
            wanted_plans.append(item)
        elif family == "losito" and "losito" in normalized_run_types:
            wanted_plans.append(item)
        elif family == "ps-losito" and "ps-losito" in normalized_run_types:
            if not selected_subject_tokens or _normalize_selector(group_key) in selected_subject_tokens:
                wanted_plans.append(item)
        elif family == "psit-loso" and "psit-loso" in normalized_run_types:
            if not selected_situation_tokens or _normalize_selector(group_key) in selected_situation_tokens:
                wanted_plans.append(item)

    if not wanted_plans:
        raise ValueError(
            f"No CV summaries selected. Received run_types={run_types}."
        )

    for summary in wanted_plans:
        print("=" * 80)
        print(f"Running CV summary: {summary['summary_label']}")
        print(f"Folds: {len(summary['folds'])}")
        print("=" * 80)

        rows = []
        for fold in summary["folds"]:
            for algorithm_name, model_mode in algorithms:
                print(
                    f"[CV] {summary['summary_label']} | fold={fold['fold_id']} | "
                    f"algorithm={algorithm_name} ({model_mode})"
                )
                row = _run_fold(
                    root=root,
                    fold=fold,
                    algorithm=algorithm_name,
                    model_mode=model_mode,
                    train_config=train_config,
                    predict_config=predict_config,
                    model=args.model,
                    fps=fps,
                    summary_label=summary["summary_slug"],
                    run_label=run_label,
                )
                rows.append(row)

        rows_with_agg = _append_aggregate_rows(rows)
        algo_suffix = "all" if len(algorithms) != 1 else algorithms[0][0]
        summary_path = Path(output_dir) / f"CV_summary_{run_label}_{summary['summary_slug']}_{algo_suffix}.csv"
        _write_summary_csv(summary_path, rows_with_agg)
        print(f"Saved CV summary: {summary_path}")


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description="Cross-validation automation processor")
    parser.add_argument(
        "--model",
        choices=["transformer_encoder", "transformer", "BERT"],
        default="transformer_encoder",
        help="Model family",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to CV YAML file")
    parser.add_argument("--run_label", type=str, default=None, help="Run namespace label used in output file names")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated algorithms: p2p_insole, soleformer",
    )
    parser.add_argument(
        "--run_types",
        type=str,
        default=None,
        help="Comma-separated CV types: LOSO, LOSitO, PS-LOSitO, PSit-LOSO",
    )
    parser.add_argument(
        "--ps_losito_subjects",
        type=str,
        default=None,
        help="Optional comma-separated subject keys to run for PS-LOSitO only",
    )
    parser.add_argument(
        "--psit_loso_situations",
        type=str,
        default=None,
        help="Optional comma-separated situation names to run for PSit-LOSO only",
    )
    parser.add_argument("--fps", type=float, default=None, help="Sampling rate for velocity-energy metrics")
    return parser
