import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


def print_config(params):
    print("<< Final Configuration Settings >>", flush=True)
    print("[Model Parameters]")
    print(f"  model_mode: {params.get('model_mode', 'n/a')}")
    print(f"  d_model: {params.get('d_model', 'n/a')}")
    print(f"  n_head: {params.get('n_head', 'n/a')}")
    print(f"  num_encoder_layer: {params.get('num_encoder_layer', 'n/a')}")
    print(f"  dropout: {params.get('dropout', 'n/a')}")
    print(f"  sequence_size: {params.get('sequence_len', 'n/a')}")
    print(f"  use_gradient_data: {params.get('use_gradient_data', False)}")
    print("\n[Training Parameters]")
    print(f"  Epochs: {params.get('num_epoch', 'n/a')}")
    print(f"  Batch Size: {params.get('batch_size', 'n/a')}")
    print("\n[Optimization Parameters]")
    print(f"  Learning Rate: {params.get('learning_rate', 'n/a')}")
    print(f"  Weight Decay: {params.get('weight_decay', 'n/a')}")
    print("\n[Loss Function Parameters]")
    print(f"  Loss Alpha: {params.get('loss_alpha', 'mse-only')}")
    print(f"  Loss Beta: {params.get('loss_beta', 'mse-only')}")
    print("\n[Other Settings]")
    print(f"  Input Dimension: {params.get('input_dim', 'n/a')}")
    print(f"  Output Dimension: {params.get('output_dim', 'n/a')}")
    print(f"  Number of Joints: {params.get('num_joints', 'n/a')}")
    print(f"  Number of Dimensions: {params.get('num_dims', 'n/a')}")
    print(f"  use_graph_pressure: {params.get('use_graph_pressure', 'n/a')}")
    print(f"  use_single_attention: {params.get('use_single_attention', 'n/a')}")
    print("---" * 20)


def format_ablation_tag(abl_id):
    if abl_id is None:
        return ""

    tag = str(abl_id).strip()
    if not tag:
        return ""

    return tag if tag.startswith("abl_id_") else f"abl_id_{tag}"


def resolve_ablation_id(config, section_name):
    section = config.get(section_name, {})
    if isinstance(section, dict):
        abl_id = section.get("abl_id", None)
        if abl_id is not None and str(abl_id).strip():
            return abl_id
    abl_id = config.get("abl_id", None)
    if abl_id is not None and str(abl_id).strip():
        return abl_id
    return None


def join_nonempty(*parts):
    values = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            values.append(text)
    return "_".join(values)


def is_repo_root(path: Path) -> bool:
    has_controller = (path / "sources" / "main.py").is_file() or (path / "main.py").is_file()
    return has_controller and (path / "notebooks").is_dir() and (path / "data").is_dir()


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if is_repo_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find repository root from {start}. Open this notebook from the project folder that contains sources/main.py, notebooks/, and data/."
    )


def initialize_notebook_runtime(start: Path | None = None):
    root = Path.cwd() if start is None else Path(start)
    if not is_repo_root(root):
        root = find_repo_root(root)
    os.chdir(root)

    python_cmd = sys.executable
    print(f"Repository root: {root}")
    print(f"Using Python: {python_cmd}")
    print(f"Python version: {sys.version}")
    return root, python_cmd


def run_cmd_streaming(cmd, cwd):
    import codecs
    from IPython.display import display

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(map(str, cmd)))

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
    )
    assert process.stdout is not None

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    progress_handle = None
    pending_cr = False

    def _update_progress(line_text):
        nonlocal progress_handle
        if not line_text:
            return
        if progress_handle is None:
            handle = display("", display_id=True)
            if handle is None:
                return
            progress_handle = handle
        progress_handle.update(line_text)

    try:
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break

            text = decoder.decode(chunk)
            for ch in text:
                if pending_cr:
                    if ch == "\n":
                        print(buffer, flush=True)
                        buffer = ""
                        pending_cr = False
                        continue
                    _update_progress(buffer.strip())
                    buffer = ""
                    pending_cr = False

                if ch == "\r":
                    pending_cr = True
                elif ch == "\n":
                    print(buffer, flush=True)
                    buffer = ""
                else:
                    buffer += ch

        trailing = decoder.decode(b"", final=True)
        if trailing:
            for ch in trailing:
                if pending_cr:
                    if ch == "\n":
                        print(buffer, flush=True)
                        buffer = ""
                        pending_cr = False
                        continue
                    _update_progress(buffer.strip())
                    buffer = ""
                    pending_cr = False

                if ch == "\r":
                    pending_cr = True
                elif ch == "\n":
                    print(buffer, flush=True)
                    buffer = ""
                else:
                    buffer += ch

        if pending_cr:
            _update_progress(buffer.strip())
            buffer = ""
            pending_cr = False

        if buffer:
            print(buffer, flush=True)
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _extract_tag(path_obj: Path, prefix: str):
    stem = path_obj.stem
    if stem.startswith(prefix + "_"):
        return stem.split("_", 1)[1]
    return None


def has_runtime_data(root: Path) -> bool:
    train_insole = list((root / "data" / "training_data" / "Insole").glob("Soles_*.txt"))
    train_skel = list((root / "data" / "training_data" / "skeleton").glob("Awinda_*.csv"))
    test_insole = list((root / "data" / "test_data" / "Insole").glob("Soles_*.txt"))
    test_skel = list((root / "data" / "test_data" / "skeleton").glob("Awinda_*.csv"))
    return bool(train_insole and train_skel and test_insole and test_skel)


def has_soleformer_training_targets(root: Path) -> bool:
    train_skel_files = list((root / "data" / "training_data" / "skeleton").glob("Awinda_*.csv"))
    target_dir = root / "data" / "clean_data" / "Awinda_targets_soleformer"
    if not train_skel_files or not target_dir.is_dir():
        return False

    for skel in train_skel_files:
        tag = _extract_tag(skel, "Awinda")
        if not tag:
            return False
        target_path = target_dir / f"AwindaTarget_{tag}.csv"
        if not target_path.is_file():
            return False
    return True


def run_preprocessing_notebook(preprocess_nb: Path, root: Path, python_cmd: str):
    if not preprocess_nb.is_file():
        raise FileNotFoundError(f"Preprocessing notebook not found: {preprocess_nb}")

    cmd = [
        python_cmd,
        "-u",
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(preprocess_nb),
    ]
    print("Running preprocessing notebook with live logs:")
    run_cmd_streaming(cmd, cwd=root)


def ensure_runtime_data_ready(root: Path, python_cmd: str, preprocess_nb: Path | None = None):
    if preprocess_nb is None:
        preprocess_nb = root / "notebooks" / "usefull_tools" / "data_preprocessing.ipynb"

    required_paths = [
        root / "data" / "clean_data",
        root / "data" / "clean_data" / "Awinda_targets_soleformer",
        root / "data" / "training_data" / "Insole",
        root / "data" / "training_data" / "skeleton",
        root / "data" / "test_data" / "Insole",
        root / "data" / "test_data" / "skeleton",
    ]

    for path in required_paths:
        path.mkdir(parents=True, exist_ok=True)

    if has_runtime_data(root) and has_soleformer_training_targets(root):
        print("Preprocessing outputs already detected in runtime folders.")
        print("Skipping re-run. Delete/refresh runtime files if you want a fresh preprocessing pass.")
        return

    preprocessing_failed = False
    try:
        run_preprocessing_notebook(preprocess_nb, root=root, python_cmd=python_cmd)
    except subprocess.CalledProcessError as exc:
        preprocessing_failed = True
        if has_runtime_data(root) and has_soleformer_training_targets(root):
            print("Warning: preprocessing notebook exited with an error, but runtime data and SoleFormer training targets are valid.")
            print(f"Continuing pipeline. Underlying command failed with exit code {exc.returncode}.")
        else:
            raise

    if not has_runtime_data(root):
        raise RuntimeError(
            "Preprocessing notebook finished but runtime train/test files are still missing. "
            "Open notebooks/usefull_tools/data_preprocessing.ipynb and inspect the last routing/validation cell."
        )

    if not has_soleformer_training_targets(root):
        raise RuntimeError(
            "Runtime data exists, but SoleFormer training target files are incomplete. "
            "Check data_preprocessing.ipynb target export and raw Awinda tab availability."
        )

    if preprocessing_failed:
        print("Proceeding with existing valid outputs after preprocessing warning.")
    else:
        print("Preprocessing completed and runtime data is ready.")


def print_csv_table(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"CSV is empty: {csv_path}")
        return

    max_cols = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    col_widths = [max(len(row[i]) for row in padded_rows) for i in range(max_cols)]

    for row in padded_rows:
        print(" | ".join(value.ljust(col_widths[i]) for i, value in enumerate(row)))


_MODE_ALLOWED_FLAGS = {
    "train": {
        "--use_cycle_loss",
        "--enable_imu_cycle_loss",
        "--enable_pressure_cycle_loss",
        "--freeze_pretrained_cycle_nets",
        "--pretrain_accelnet",
        "--pretrain_pressnet",
        "--pretrain_epochs",
        "--pretrain_learning_rate",
        "--accelnet_pretrained_path",
        "--pressnet_pretrained_path",
        "--use_time_feature",
        "--use_gradient_data",
        "--grad_window_length",
        "--grad_polyorder",
        "--grad_smooth_grad1",
        "--include_target_positions",
        "--include_target_joint_angles",
        "--joint_angles_tab_suffix",
        "--smoothing_sigma",
        "--soleformer_use_graph_pressure",
        "--soleformer_use_single_attention",
    },
    "predict": {
        "--use_time_feature",
        "--use_gradient_data",
        "--grad_window_length",
        "--grad_polyorder",
        "--grad_smooth_grad1",
        "--smoothing_sigma",
        "--max_windows",
        "--soleformer_use_graph_pressure",
        "--soleformer_use_single_attention",
    },
    "visual": set(),
}


def normalize_abl_id(value):
    if value is None:
        return ""
    return str(value).strip()


def find_ablation_row(abl_id, csv_path: Path):
    normalized_id = normalize_abl_id(abl_id)
    if not normalized_id:
        return None

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if normalize_abl_id(row.get("abl_id")) == normalized_id:
            return row

    raise ValueError(f"No ablation row found for abl_id={normalized_id} in {csv_path}")


def ablation_row_label(row):
    if not row:
        return ""
    return f"abl_id={row.get('abl_id', '').strip()} | {row.get('Category', '').strip()} | {row.get('Ablation', '').strip()}"


def extract_cli_options_from_train_command(command_line):
    tokens = shlex.split(command_line)
    if "train" not in tokens:
        return []
    return tokens[tokens.index("train") + 1 :]


def merge_missing_flags(base_cmd, extra_options, allowed_flags):
    merged_cmd = list(base_cmd)
    existing_flags = {token for token in merged_cmd if isinstance(token, str) and token.startswith("--")}
    added_tokens = []

    i = 0
    while i < len(extra_options):
        token = extra_options[i]
        if not token.startswith("--"):
            i += 1
            continue

        flag = token
        has_value = i + 1 < len(extra_options) and not extra_options[i + 1].startswith("--")
        value = extra_options[i + 1] if has_value else None
        step = 2 if has_value else 1

        if flag in allowed_flags and flag not in existing_flags:
            merged_cmd.append(flag)
            added_tokens.append(flag)
            if has_value:
                merged_cmd.append(value)
                added_tokens.append(value)
            existing_flags.add(flag)

        i += step

    return merged_cmd, added_tokens


def add_ablation_flags(mode, base_cmd, abl_id, csv_path: Path):
    row = find_ablation_row(abl_id, csv_path)
    if row is None:
        return list(base_cmd), [], None

    command_line = (row.get("CommandLine") or "").strip()
    if not command_line:
        return list(base_cmd), [], row

    extra_options = extract_cli_options_from_train_command(command_line)
    allowed_flags = _MODE_ALLOWED_FLAGS.get(mode, set())
    merged_cmd, added_tokens = merge_missing_flags(base_cmd, extra_options, allowed_flags)
    return merged_cmd, added_tokens, row
