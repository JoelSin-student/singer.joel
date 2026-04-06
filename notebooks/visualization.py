# Visualization processor
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from notebooks.loader import load_config


def _find_xyz_columns(df):
    cols = [str(c) for c in df.columns]

    indexed = {}
    for col in cols:
        m = re.match(r'^(?:pos::)?([XYZ])\.(\d+)$', col)
        if not m:
            continue
        axis = m.group(1)
        idx = int(m.group(2))
        indexed.setdefault(idx, {})[axis] = col

    if indexed:
        ordered = []
        for idx in sorted(indexed.keys()):
            axes = indexed[idx]
            if all(k in axes for k in ('X', 'Y', 'Z')):
                ordered.append((axes['X'], axes['Y'], axes['Z']))
        if ordered:
            return ordered

    named = {}
    for col in cols:
        m = re.match(r'^(?:pos::)?(.+?)\s*([XYZxyz])$', col)
        if not m:
            continue
        name = m.group(1).strip()
        axis = m.group(2).upper()
        named.setdefault(name, {})[axis] = col

    ordered = []
    for name in sorted(named.keys()):
        axes = named[name]
        if all(k in axes for k in ('X', 'Y', 'Z')):
            ordered.append((axes['X'], axes['Y'], axes['Z']))
    return ordered


def _process_skeleton_data(df):
    xyz_triplets = _find_xyz_columns(df)
    if not xyz_triplets:
        raise ValueError(
            "Could not detect 3D coordinate columns. Expected columns like X.i/Y.i/Z.i "
            "or pos::X.i/pos::Y.i/pos::Z.i."
        )

    frames_data = []
    for _, row in df.iterrows():
        x_positions, y_positions, z_positions = [], [], []
        for x_col, y_col, z_col in xyz_triplets:
            x = row.get(x_col)
            y = row.get(y_col)
            z = row.get(z_col)
            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                x_positions.append(float(x))
                y_positions.append(float(y))
                z_positions.append(float(z))

        if x_positions:
            frames_data.append({'x': x_positions, 'y': y_positions, 'z': z_positions})

    return frames_data


def _joint_xyz(frame, idx):
    if idx >= len(frame['x']):
        return [], [], []
    return [frame['x'][idx]], [frame['z'][idx]], [frame['y'][idx]]


def _all_bones_xyz(frame, bones):
    xs, ys, zs = [], [], []
    for start, end in bones:
        if start < len(frame['x']) and end < len(frame['x']):
            xs.extend([frame['x'][start], frame['x'][end], None])
            ys.extend([frame['z'][start], frame['z'][end], None])
            zs.extend([frame['y'][start], frame['y'][end], None])
    return xs, ys, zs


def _bone_xyz(frame, start, end):
    if start >= len(frame['x']) or end >= len(frame['x']):
        return [], [], []
    return [frame['x'][start], frame['x'][end]], [frame['z'][start], frame['z'][end]], [frame['y'][start], frame['y'][end]]


def _create_frame_traces(frame_real, frame_pred, bones, joint_indices, showlegend=False):
    traces = []

    traces.append(
        go.Scatter3d(
            x=frame_real['x'],
            y=frame_real['z'],
            z=frame_real['y'],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.8),
            name='Real Joints (All)',
            showlegend=showlegend,
        )
    )
    real_bones_x, real_bones_y, real_bones_z = _all_bones_xyz(frame_real, bones)
    traces.append(
        go.Scatter3d(
            x=real_bones_x,
            y=real_bones_y,
            z=real_bones_z,
            mode='lines',
            line=dict(color='red', width=2),
            name='Real Bones (All)',
            showlegend=showlegend,
        )
    )

    traces.append(
        go.Scatter3d(
            x=frame_pred['x'],
            y=frame_pred['z'],
            z=frame_pred['y'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.8),
            name='Predicted Joints (All)',
            showlegend=showlegend,
        )
    )
    pred_bones_x, pred_bones_y, pred_bones_z = _all_bones_xyz(frame_pred, bones)
    traces.append(
        go.Scatter3d(
            x=pred_bones_x,
            y=pred_bones_y,
            z=pred_bones_z,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Predicted Bones (All)',
            showlegend=showlegend,
        )
    )

    for joint_idx in joint_indices:
        real_x, real_y, real_z = _joint_xyz(frame_real, joint_idx)
        traces.append(
            go.Scatter3d(
                x=real_x,
                y=real_y,
                z=real_z,
                mode='markers',
                marker=dict(size=6, color='red', opacity=0.9),
                name=f'Real Joint {joint_idx}',
                visible='legendonly',
                showlegend=showlegend,
            )
        )

        pred_x, pred_y, pred_z = _joint_xyz(frame_pred, joint_idx)
        traces.append(
            go.Scatter3d(
                x=pred_x,
                y=pred_y,
                z=pred_z,
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.9),
                name=f'Predicted Joint {joint_idx}',
                visible='legendonly',
                showlegend=showlegend,
            )
        )

    for bone_idx, (start, end) in enumerate(bones):
        real_bx, real_by, real_bz = _bone_xyz(frame_real, start, end)
        traces.append(
            go.Scatter3d(
                x=real_bx,
                y=real_by,
                z=real_bz,
                mode='lines',
                line=dict(color='red', width=3),
                name=f'Real Bone {bone_idx} ({start}-{end})',
                visible='legendonly',
                showlegend=showlegend,
            )
        )

        pred_bx, pred_by, pred_bz = _bone_xyz(frame_pred, start, end)
        traces.append(
            go.Scatter3d(
                x=pred_bx,
                y=pred_by,
                z=pred_bz,
                mode='lines',
                line=dict(color='blue', width=3),
                name=f'Predicted Bone {bone_idx} ({start}-{end})',
                visible='legendonly',
                showlegend=showlegend,
            )
        )

    return traces


def _resolve_files(config, args):
    data_path = config['location']['data_path']
    skeleton_dir = Path(data_path) / 'skeleton'
    
    # CLI arguments override config values.
    real_file_arg = (
        (args.real_file if args and args.real_file else None)
        or config['visual'].get('real_file', None)
        or config['visual'].get('ground_truth_file', None)
    )
    pred_file_arg = (
        (args.pred_file if args and args.pred_file else None)
        or config['visual'].get('pred_file', None)
        or config['visual'].get('prediction_file', None)
    )
    model_mode = str(config['visual'].get('model_mode', 'simple_seq2seq')).lower()

    if real_file_arg:
        file_path_real = Path(real_file_arg)
    else:
        real_candidates = sorted(skeleton_dir.glob('Awinda_*.csv'))
        if not real_candidates:
            raise FileNotFoundError(f"No Awinda_*.csv file found in {skeleton_dir}")
        file_path_real = real_candidates[0]

    if pred_file_arg:
        file_path_predict = Path(pred_file_arg)
    else:
        pred_dir = Path(".") / "results" / "output"
        mode_candidates = list(pred_dir.glob(f'Predicted_skeleton*{model_mode}*.csv'))
        all_candidates = list(pred_dir.glob('Predicted_skeleton*.csv'))
        candidates = mode_candidates if mode_candidates else all_candidates
        if not candidates:
            raise FileNotFoundError('No Predicted_skeleton*.csv file found in results/output')
        # When multiple files exist, use the most recently modified
        file_path_predict = max(candidates, key=lambda p: p.stat().st_mtime)

    if not file_path_real.is_file():
        raise FileNotFoundError(f"Real skeleton file not found: {file_path_real}")
    if not file_path_predict.is_file():
        raise FileNotFoundError(f"Prediction file not found: {file_path_predict}")

    return file_path_real, file_path_predict


def _extract_tag_from_real_file(file_path_real):
    stem = file_path_real.stem
    if stem.startswith('Awinda_'):
        return stem[len('Awinda_'):]
    return stem


def _extract_tag_from_pred_file(file_path_predict, model_mode):
    stem = file_path_predict.stem
    if stem.startswith('Predicted_skeleton_'):
        stem = stem[len('Predicted_skeleton_'):]
    suffix = f'_{model_mode}'
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    return stem


def _build_input_tag(*tags):
    unique_tags = []
    for tag in tags:
        tag_str = str(tag).strip()
        if tag_str and tag_str not in unique_tags:
            unique_tags.append(tag_str)
    if not unique_tags:
        return 'unknown'
    return '__'.join(unique_tags)





def start(args=None):
    config = load_config(args, args.config if args else None, args.model if args else None)
    model_mode = str(config['visual'].get('model_mode', 'simple_seq2seq')).lower()

    file_path_real, file_path_predict = _resolve_files(config, args)
    print(f"Using real skeleton file: {file_path_real}")
    print(f"Using predicted skeleton file: {file_path_predict}")

    # CLI arguments override config values.
    start_frame = int((args.start_frame if args and args.start_frame is not None else None) or config['visual'].get('start_frame', 0))
    step = int((args.step if args and args.step is not None else None) or config['visual'].get('step', 3))
    output_dir_config = (
        (args.output_html if args and args.output_html else None)
        or config['visual'].get('output_html', None)
        or os.path.join(".", "results", "animation")
    )
    
    real_tag = _extract_tag_from_real_file(file_path_real)
    pred_tag = _extract_tag_from_pred_file(file_path_predict, model_mode)
    input_tag = _build_input_tag(real_tag, pred_tag)

    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (4, 7), (4, 11), (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14),
        (0, 15), (0, 19), (15, 19),
        (15, 16), (16, 17), (17, 18), (19, 20), (20, 21), (21, 22),
    ]

    df_real = pd.read_csv(file_path_real)
    df_pred = pd.read_csv(file_path_predict)
    print(f"Real data shape: {df_real.shape}")
    print(f"Pred data shape: {df_pred.shape}")

    if 'Frame' in df_real.columns and 'Frame' in df_pred.columns:
        common_frames = sorted(set(df_real['Frame'].tolist()) & set(df_pred['Frame'].tolist()))
        if common_frames:
            df_real = df_real[df_real['Frame'].isin(common_frames)].sort_values('Frame').reset_index(drop=True)
            df_pred = df_pred[df_pred['Frame'].isin(common_frames)].sort_values('Frame').reset_index(drop=True)

    frames_data_real = _process_skeleton_data(df_real)
    frames_data_pred = _process_skeleton_data(df_pred)

    end_frame = min(len(frames_data_real), len(frames_data_pred))
    frames_data_real = frames_data_real[start_frame:end_frame:step]
    frames_data_pred = frames_data_pred[start_frame:end_frame:step]
    if not frames_data_real or not frames_data_pred:
        raise ValueError('No visualization frames available after slicing. Check start_frame/step settings.')

    max_joint_count = max(
        max((len(frame['x']) for frame in frames_data_real), default=0),
        max((len(frame['x']) for frame in frames_data_pred), default=0),
    )
    joint_indices = list(range(max_joint_count))

    fig = go.Figure()
    initial_traces = _create_frame_traces(
        frames_data_real[0],
        frames_data_pred[0],
        bones,
        joint_indices,
        showlegend=True,
    )
    for trace in initial_traces:
        fig.add_trace(trace)

    frames = [
        go.Frame(data=_create_frame_traces(real, pred, bones, joint_indices, showlegend=False), name=f'frame{i}')
        for i, (real, pred) in enumerate(zip(frames_data_real, frames_data_pred))
    ]
    fig.frames = frames

    fig.update_layout(
        title='3D Skeleton Animation: Real (Red) vs Prediction (Blue)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000,
        height=1000,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1.0,
            yanchor='top',
            xanchor='left',
            itemsizing='constant',
        ),
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'},
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top',
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {'font': {'size': 20}, 'prefix': 'Frame:', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': str(k),
                    'method': 'animate',
                }
                for k, f in enumerate(frames)
            ],
        }],
    )

    # Treat output_dir_config as a directory path; if a file path is given, use its parent.
    output_dir = Path(output_dir_config)
    if output_dir.suffix:  # Has file extension, so extract parent directory
        output_dir = output_dir.parent
    
    output_file = output_dir / f'Animation_{input_tag}_{model_mode}.html'
    os.makedirs(output_dir, exist_ok=True)
    html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_str)

    print(f"num frames: {len(frames)}")
    print(f"Visualization successful. Saved to: {output_file}")


def get_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help, description='Visualization Processor')

    parser.add_argument('--model', choices=['transformer_encoder', 'transformer', 'BERT'], default='transformer_encoder')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML file')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_mode', type=str, default=None, choices=['original', 'simple_seq2seq', 'soleformer'])
    parser.add_argument('--pred_file', '--prediction_file', dest='pred_file', type=str, default=None, help='Path to predicted skeleton CSV')
    parser.add_argument('--real_file', '--ground_truth_file', dest='real_file', type=str, default=None, help='Path to ground-truth skeleton CSV')
    parser.add_argument('--start_frame', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--output_html', type=str, default=None)

    return parser