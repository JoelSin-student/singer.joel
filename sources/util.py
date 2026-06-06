import csv
import json
import os
import re
import shlex
import shutil
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
    print(f"  pose_loss_mode: {params.get('pose_loss_mode', 'both')}")
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


def _in_notebook_runtime():
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def run_cmd_streaming(cmd, cwd, env_overrides=None, stream_mode="auto"):
    import codecs

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        for key, value in env_overrides.items():
            env[str(key)] = str(value)
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

    mode = str(stream_mode or "auto").strip().lower()
    if mode not in {"auto", "raw", "notebook"}:
        raise ValueError("stream_mode must be one of: auto, raw, notebook")
    if mode == "auto":
        mode = "notebook" if _in_notebook_runtime() else "raw"

    progress_handle = None
    line_buffer = ""

    def _write_raw(text):
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _write_notebook(text):
        nonlocal progress_handle, line_buffer

        if not text:
            return

        for ch in text:
            if ch == "\r":
                if line_buffer.strip():
                    try:
                        from IPython.display import display

                        if progress_handle is None:
                            progress_handle = display("", display_id=True)
                        if progress_handle is not None:
                            progress_handle.update(line_buffer.strip())
                    except Exception:
                        print(line_buffer.strip(), end="\r", flush=True)
                line_buffer = ""
            elif ch == "\n":
                print(line_buffer, flush=True)
                line_buffer = ""
            else:
                line_buffer += ch

    try:
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break

            text = decoder.decode(chunk)
            if mode == "raw":
                _write_raw(text)
            else:
                _write_notebook(text)

        trailing = decoder.decode(b"", final=True)
        if mode == "raw":
            _write_raw(trailing)
        else:
            _write_notebook(trailing)
            if line_buffer:
                print(line_buffer, flush=True)
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


def _normalize_requested_key(value, split: str):
    key = str(value).strip()
    if not key:
        return ""

    split_suffix = f"_{split}"
    if key.lower().endswith(split_suffix):
        key = key[: -len(split_suffix)]
    return key


def has_runtime_data(root: Path) -> bool:
    train_insole = list((root / "data" / "training_data" / "Insole").glob("Soles_*.txt"))
    train_skel = list((root / "data" / "training_data" / "skeleton").glob("Awinda_*.csv"))
    test_insole = list((root / "data" / "test_data" / "Insole").glob("Soles_*.txt"))
    test_skel = list((root / "data" / "test_data" / "skeleton").glob("Awinda_*.csv"))
    return bool(train_insole and train_skel and test_insole and test_skel)


def run_preprocessing_notebook(preprocess_nb: Path, root: Path, python_cmd: str, env_overrides=None):
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
    effective_env = {
        # Suppress noisy pydevd frozen-module validation warnings emitted by ipykernel.
        "PYDEVD_DISABLE_FILE_VALIDATION": "1",
    }
    if env_overrides:
        effective_env.update(env_overrides)

    run_cmd_streaming(cmd, cwd=root, env_overrides=effective_env)
    _print_preprocessing_log_summary(preprocess_nb)


def _iter_notebook_output_lines(notebook_path: Path):
    try:
        payload = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not parse preprocessing notebook outputs: {exc}")
        return

    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            text = output.get("text")
            if isinstance(text, str):
                for line in text.splitlines():
                    yield line
            elif isinstance(text, list):
                for chunk in text:
                    for line in str(chunk).splitlines():
                        yield line


def _print_preprocessing_log_summary(preprocess_nb: Path):
    """Print key preprocessing logs captured in executed notebook outputs."""
    patterns = {
        "anomaly": ["[special-fix]", "anomaly removed", "synthetic block", "post-gap shift"],
        "start_offset": ["[global-trim]", "rows trimmed", "delta/refs"],
        "pair_tail": ["[pair tail-trim]", "Pair tail-alignment summary:"],
    }

    grouped = {key: [] for key in patterns}
    for line in _iter_notebook_output_lines(preprocess_nb):
        normalized = line.strip()
        if not normalized:
            continue
        for key, tokens in patterns.items():
            if any(token in normalized for token in tokens):
                grouped[key].append(normalized)

    print("\nPreprocessing log summary (from notebook outputs):")
    labels = {
        "anomaly": "Anomaly removal",
        "start_offset": "Start offset trimming",
        "pair_tail": "Pair tail-trim",
    }
    for key in ["anomaly", "start_offset", "pair_tail"]:
        lines = grouped[key]
        print(f"  {labels[key]}:")
        if not lines:
            print("    - no matching log lines found")
            continue

        for item in lines:
            print(f"    - {item}")


def ensure_runtime_data_ready(root: Path, python_cmd: str, preprocess_nb: Path | None = None):
    if preprocess_nb is None:
        preprocess_nb = root / "notebooks" / "usefull_tools" / "data_preprocessing.ipynb"

    required_paths = [
        root / "data" / "clean_data",
        root / "data" / "training_data" / "Insole",
        root / "data" / "training_data" / "skeleton",
        root / "data" / "test_data" / "Insole",
        root / "data" / "test_data" / "skeleton",
    ]

    for path in required_paths:
        path.mkdir(parents=True, exist_ok=True)

    if has_runtime_data(root):
        print("Preprocessing outputs already detected in runtime folders.")
        print("Skipping re-run. Delete/refresh runtime files if you want a fresh preprocessing pass.")
        return

    preprocessing_failed = False
    try:
        run_preprocessing_notebook(preprocess_nb, root=root, python_cmd=python_cmd)
    except subprocess.CalledProcessError as exc:
        preprocessing_failed = True
        if has_runtime_data(root):
            print("Warning: preprocessing notebook exited with an error, but runtime data is valid.")
            print(f"Continuing pipeline. Underlying command failed with exit code {exc.returncode}.")
        else:
            raise

    if not has_runtime_data(root):
        raise RuntimeError(
            "Preprocessing notebook finished but runtime train/test files are still missing. "
            "Open notebooks/usefull_tools/data_preprocessing.ipynb and inspect the last routing/validation cell."
        )

    if preprocessing_failed:
        print("Proceeding with existing valid outputs after preprocessing warning.")
    else:
        print("Preprocessing completed and runtime data is ready.")


def _first_existing(paths):
    for path in paths:
        if Path(path).is_file():
            return Path(path)
    return None


def _resolve_insole_source(raw_insole_dir: Path, key: str, split: str) -> Path:
    split_dir = "Soles_training" if split == "training" else "Soles_test"
    candidates = [
        raw_insole_dir / f"Soles_{key}.txt",
        raw_insole_dir / f"Soles_{key}_{split}.txt",
        raw_insole_dir / split_dir / f"Soles_{key}.txt",
        raw_insole_dir / split_dir / f"Soles_{key}_{split}.txt",
    ]
    src = _first_existing(candidates)
    if src is None:
        searched = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Insole source not found for key '{key}' ({split}). Checked:\n  - {searched}"
        )
    return src


def _resolve_awinda_source(raw_awinda_dir: Path, key: str, split: str) -> Path:
    split_dir = "Awinda_training" if split == "training" else "Awinda_test"
    candidates = [
        raw_awinda_dir / f"Awinda_{key}.csv",
        raw_awinda_dir / f"Awinda_{key}.xlsx",
        raw_awinda_dir / f"Awinda_{key}_{split}.csv",
        raw_awinda_dir / f"Awinda_{key}_{split}.xlsx",
        raw_awinda_dir / split_dir / f"Awinda_{key}.csv",
        raw_awinda_dir / split_dir / f"Awinda_{key}.xlsx",
        raw_awinda_dir / split_dir / f"Awinda_{key}_{split}.csv",
        raw_awinda_dir / split_dir / f"Awinda_{key}_{split}.xlsx",
    ]
    src = _first_existing(candidates)
    if src is None:
        searched = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Awinda source not found for key '{key}' ({split}). Checked:\n  - {searched}"
        )
    return src


def _export_awinda_csv(src_path: Path, dst_csv_path: Path):
    src_path = Path(src_path)
    dst_csv_path = Path(dst_csv_path)

    if src_path.suffix.lower() == ".csv":
        shutil.copy2(src_path, dst_csv_path)
        return

    if src_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Unsupported Awinda source extension: {src_path.suffix}")

    import pandas as pd

    xls = pd.ExcelFile(src_path)
    sheet_name = "Segment Position" if "Segment Position" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.to_csv(dst_csv_path, index=False, sep=",", decimal=".")


def _clear_runtime_top_level(runtime_train_insole_dir: Path, runtime_train_skel_dir: Path, runtime_test_insole_dir: Path, runtime_test_skel_dir: Path):
    for path in runtime_train_insole_dir.glob("Soles_*.txt"):
        if path.is_file():
            path.unlink()
    for path in runtime_train_skel_dir.glob("Awinda_*.csv"):
        if path.is_file():
            path.unlink()
    for path in runtime_test_insole_dir.glob("Soles_*.txt"):
        if path.is_file():
            path.unlink()
    for path in runtime_test_skel_dir.glob("Awinda_*.csv"):
        if path.is_file():
            path.unlink()


def _route_split(keys, split: str, raw_awinda_dir: Path, raw_insole_dir: Path, runtime_train_skel_dir: Path, runtime_train_insole_dir: Path, runtime_test_skel_dir: Path, runtime_test_insole_dir: Path):
    if split == "training":
        dst_insole_dir = runtime_train_insole_dir
        dst_skel_dir = runtime_train_skel_dir
    else:
        dst_insole_dir = runtime_test_insole_dir
        dst_skel_dir = runtime_test_skel_dir

    routed = []
    for key in keys:
        normalized_key = _normalize_requested_key(key, split)
        if not normalized_key:
            continue

        insole_src = _resolve_insole_source(raw_insole_dir, normalized_key, split)
        awinda_src = _resolve_awinda_source(raw_awinda_dir, normalized_key, split)

        insole_dst = dst_insole_dir / f"Soles_{normalized_key}_{split}.txt"
        awinda_dst = dst_skel_dir / f"Awinda_{normalized_key}_{split}.csv"

        shutil.copy2(insole_src, insole_dst)
        _export_awinda_csv(awinda_src, awinda_dst)

        routed.append(
            {
                "key": normalized_key,
                "split": split,
                "insole_src": str(insole_src),
                "insole_dst": str(insole_dst),
                "awinda_src": str(awinda_src),
                "awinda_dst": str(awinda_dst),
            }
        )

    return routed


def _collect_runtime_tags(insole_dir: Path, skel_dir: Path):
    insole_tags = {
        path.stem.split("_", 1)[1]
        for path in insole_dir.glob("Soles_*.txt")
        if path.is_file() and "_" in path.stem
    }
    skel_tags = {
        path.stem.split("_", 1)[1]
        for path in skel_dir.glob("Awinda_*.csv")
        if path.is_file() and "_" in path.stem
    }
    return insole_tags, skel_tags


def _validate_runtime_exact_keys(runtime_train_insole_dir: Path, runtime_train_skel_dir: Path, runtime_test_insole_dir: Path, runtime_test_skel_dir: Path, train_files, test_files):
    expected_train = {
        f"{_normalize_requested_key(item, 'training')}_training"
        for item in train_files
        if _normalize_requested_key(item, 'training')
    }
    expected_test = {
        f"{_normalize_requested_key(item, 'test')}_test"
        for item in test_files
        if _normalize_requested_key(item, 'test')
    }

    train_insole_tags, train_skel_tags = _collect_runtime_tags(runtime_train_insole_dir, runtime_train_skel_dir)
    test_insole_tags, test_skel_tags = _collect_runtime_tags(runtime_test_insole_dir, runtime_test_skel_dir)

    if train_insole_tags != expected_train or train_skel_tags != expected_train:
        raise ValueError(
            "Runtime training folders do not exactly match train_files. "
            f"Expected: {sorted(expected_train)} | "
            f"Insole: {sorted(train_insole_tags)} | Skeleton: {sorted(train_skel_tags)}"
        )

    if test_insole_tags != expected_test or test_skel_tags != expected_test:
        raise ValueError(
            "Runtime test folders do not exactly match test_files. "
            f"Expected: {sorted(expected_test)} | "
            f"Insole: {sorted(test_insole_tags)} | Skeleton: {sorted(test_skel_tags)}"
        )


def _check_cache_coverage(cache_root: Path, train_files, test_files):
    cache_root = Path(cache_root)
    missing = []

    for split, keys in (("training", train_files), ("test", test_files)):
        for item in keys:
            key = _normalize_requested_key(item, split)
            if not key:
                continue

            try:
                _resolve_insole_source(cache_root, key, split)
            except FileNotFoundError as exc:
                missing.append(f"[Insole][{split}] {key}: {exc}")

            try:
                _resolve_awinda_source(cache_root, key, split)
            except FileNotFoundError as exc:
                missing.append(f"[Awinda][{split}] {key}: {exc}")

    return missing


def _verify_cache_covers_all_top_level_raw(cache_root: Path, root: Path):
    """Validate that clean_data covers all top-level raw capture files.

    This intentionally checks only top-level files under raw_data/Awinda and
    raw_data/Insoles, matching the project convention used by main.ipynb.
    """
    cache_root = Path(cache_root)
    root = Path(root)

    raw_awinda_dir = root / "data" / "raw_data" / "Awinda"
    raw_insole_dir = root / "data" / "raw_data" / "Insoles"

    if not raw_awinda_dir.is_dir():
        raise FileNotFoundError(f"Raw Awinda folder not found: {raw_awinda_dir}")
    if not raw_insole_dir.is_dir():
        raise FileNotFoundError(f"Raw Insole folder not found: {raw_insole_dir}")

    # Top-level capture files only; ignore extracted/synchronized tab artifacts in subfolders.
    raw_awinda_sources = sorted(
        [
            p for p in raw_awinda_dir.iterdir()
            if p.is_file() and p.name.startswith("Awinda_") and p.suffix.lower() in {".xlsx", ".csv"}
        ],
        key=lambda p: p.name,
    )
    raw_insole_sources = sorted(
        [
            p for p in raw_insole_dir.iterdir()
            if p.is_file() and p.name.startswith("Soles_") and p.suffix.lower() == ".txt"
        ],
        key=lambda p: p.name,
    )

    if not raw_awinda_sources:
        raise FileNotFoundError(f"No top-level Awinda capture files found in {raw_awinda_dir}")
    if not raw_insole_sources:
        raise FileNotFoundError(f"No top-level Insole capture files found in {raw_insole_dir}")

    missing = []

    for src in raw_awinda_sources:
        tag = _extract_tag(src, "Awinda")
        if not tag:
            continue
        expected_cache_path = cache_root / f"Awinda_{tag}.csv"
        if not expected_cache_path.is_file():
            missing.append(f"Awinda missing in clean_data: expected {expected_cache_path.name} from {src.name}")

    for src in raw_insole_sources:
        expected_cache_path = cache_root / src.name
        if not expected_cache_path.is_file():
            missing.append(f"Insole missing in clean_data: expected {expected_cache_path.name} from {src.name}")

    if missing:
        details = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "clean_data does not fully cover top-level raw capture files after preprocessing notebook execution."
            f"\ncache_root: {cache_root}"
            f"\nMissing entries:{details}"
        )


def _available_cache_tags(cache_root: Path):
    cache_root = Path(cache_root)
    insole_tags = sorted(
        path.stem.split("_", 1)[1]
        for path in cache_root.glob("Soles_*.txt")
        if path.is_file() and "_" in path.stem
    )
    awinda_tags = sorted(
        path.stem.split("_", 1)[1]
        for path in cache_root.glob("Awinda_*.csv")
        if path.is_file() and "_" in path.stem
    )
    return insole_tags, awinda_tags


def prepare_runtime_data_from_raw(
    root: Path,
    train_files,
    test_files,
    run_full_preprocessing: bool = False,
    python_cmd: str | None = None,
):
    root = Path(root)

    raw_awinda_dir = root / "data" / "raw_data" / "Awinda"
    raw_insole_dir = root / "data" / "raw_data" / "Insoles"

    runtime_train_skel_dir = root / "data" / "training_data" / "skeleton"
    runtime_train_insole_dir = root / "data" / "training_data" / "Insole"
    runtime_test_skel_dir = root / "data" / "test_data" / "skeleton"
    runtime_test_insole_dir = root / "data" / "test_data" / "Insole"

    for folder in [
        runtime_train_skel_dir,
        runtime_train_insole_dir,
        runtime_test_skel_dir,
        runtime_test_insole_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    _clear_runtime_top_level(
        runtime_train_insole_dir,
        runtime_train_skel_dir,
        runtime_test_insole_dir,
        runtime_test_skel_dir,
    )

    train_routing = _route_split(
        train_files,
        "training",
        raw_awinda_dir,
        raw_insole_dir,
        runtime_train_skel_dir,
        runtime_train_insole_dir,
        runtime_test_skel_dir,
        runtime_test_insole_dir,
    )
    test_routing = _route_split(
        test_files,
        "test",
        raw_awinda_dir,
        raw_insole_dir,
        runtime_train_skel_dir,
        runtime_train_insole_dir,
        runtime_test_skel_dir,
        runtime_test_insole_dir,
    )

    print("Runtime routing completed from raw_data using train_files/test_files.")
    print(f"  Training pairs routed: {len(train_routing)}")
    print(f"  Test pairs routed: {len(test_routing)}")

    if train_routing:
        print("\nTraining routes:")
        for row in train_routing:
            print(f"  - {row['key']} -> {row['awinda_dst']} | {row['insole_dst']}")

    if test_routing:
        print("\nTest routes:")
        for row in test_routing:
            print(f"  - {row['key']} -> {row['awinda_dst']} | {row['insole_dst']}")

    if run_full_preprocessing:
        ensure_runtime_data_ready(root, python_cmd or sys.executable)
    else:
        print("\nSkipped full preprocessing notebook execution (run_full_preprocessing=False).")
        print("If you plan to train SoleFormer on new keys, set run_full_preprocessing=True at least once.")

    return {
        "train_routing": train_routing,
        "test_routing": test_routing,
    }


def _pre_split_cache_root(root: Path) -> Path:
    # Single source of truth for pre-split preprocessed files.
    return Path(root) / "data" / "clean_data"


def rebuild_pre_split_cache_from_raw(root: Path, python_cmd: str | None = None):
    root = Path(root)
    clean_root = _pre_split_cache_root(root)
    preprocess_nb = root / "notebooks" / "usefull_tools" / "data_preprocessing.ipynb"
    python_cmd = python_cmd or sys.executable

    print("Stage 1/2: running preprocessing notebook to rebuild clean_data...")
    run_preprocessing_notebook(
        preprocess_nb=preprocess_nb,
        root=root,
        python_cmd=python_cmd,
        env_overrides={"KICKCAP_PREPROCESS_MODE": "clean_data"},
    )

    print("Stage 2/2: validating clean_data covers all top-level raw files before runtime routing...")
    if not clean_root.is_dir():
        raise FileNotFoundError(f"clean_data folder not found after notebook execution: {clean_root}")
    if not any(clean_root.glob("Awinda_*.csv")) or not any(clean_root.glob("Soles_*.txt")):
        raise RuntimeError(
            "Notebook preprocessing completed but clean_data does not contain expected Awinda/Soles outputs. "
            "Check notebooks/usefull_tools/data_preprocessing.ipynb output cells."
        )
    _verify_cache_covers_all_top_level_raw(clean_root, root)
    return clean_root


def populate_pre_split_cache(root: Path, source_root: Path | None = None):
    # Backward-compatible no-op now that clean_data is the canonical pre-split cache.
    root = Path(root)
    source_root = Path(source_root) if source_root is not None else root / "data" / "clean_data"
    cache_root = _pre_split_cache_root(root)

    if not source_root.is_dir():
        raise FileNotFoundError(f"Preprocessed source folder not found: {source_root}")

    if source_root != cache_root:
        raise ValueError(
            "populate_pre_split_cache no longer supports copying into a second cache location. "
            f"Use clean_data directly: expected source_root={cache_root}, got {source_root}."
        )

    skeleton_count = len(list(cache_root.glob("Awinda_*.csv")))
    insole_count = len(list(cache_root.glob("Soles_*.txt")))

    print(f"Using clean_data as pre-split cache source: {cache_root}")
    print(f"  available skeleton files: {skeleton_count}")
    print(f"  available insole files: {insole_count}")
    return cache_root


def prepare_runtime_data_from_pre_split_cache(
    root: Path,
    train_files,
    test_files,
):
    """Route preprocessed files from clean_data into runtime train/test folders.

    This function is read-only with respect to clean_data: it never rebuilds it.
    Call rebuild_pre_split_cache_from_raw() first if clean_data needs filling.
    """
    root = Path(root)

    cache_root = _pre_split_cache_root(root)
    if not cache_root.is_dir() or not any(cache_root.glob("Awinda_*.csv")) or not any(cache_root.glob("Soles_*.txt")):
        raise FileNotFoundError(
            f"clean_data is empty or missing: {cache_root}\n"
            "Run clean_data filling first (RUN_CLEAN_DATA_FILL = True)."
        )

    raw_awinda_dir = cache_root
    raw_insole_dir = cache_root

    runtime_train_skel_dir = root / "data" / "training_data" / "skeleton"
    runtime_train_insole_dir = root / "data" / "training_data" / "Insole"
    runtime_test_skel_dir = root / "data" / "test_data" / "skeleton"
    runtime_test_insole_dir = root / "data" / "test_data" / "Insole"

    train_set = {_normalize_requested_key(item, "training") for item in train_files if _normalize_requested_key(item, "training")}
    test_set = {_normalize_requested_key(item, "test") for item in test_files if _normalize_requested_key(item, "test")}
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"train_files and test_files must be disjoint. Overlap: {overlap}")

    missing_cache_entries = _check_cache_coverage(cache_root, train_files, test_files)
    if missing_cache_entries:
        insole_tags, awinda_tags = _available_cache_tags(cache_root)
        insole_preview = ", ".join(insole_tags[:15]) if insole_tags else "<none>"
        awinda_preview = ", ".join(awinda_tags[:15]) if awinda_tags else "<none>"
        if len(insole_tags) > 15:
            insole_preview += f", ... (+{len(insole_tags) - 15} more)"
        if len(awinda_tags) > 15:
            awinda_preview += f", ... (+{len(awinda_tags) - 15} more)"

        details = "\n  - " + "\n  - ".join(missing_cache_entries)
        raise FileNotFoundError(
            "clean_data is missing one or more requested train/test pairs. "
            "This function enforces a clean_data-first workflow: preprocess -> route/rename.\n"
            f"clean_data root: {cache_root}\n"
            f"Requested train keys: {sorted(train_set)}\n"
            f"Requested test keys: {sorted(test_set)}\n"
            f"Available insole tags in clean_data (preview): {insole_preview}\n"
            f"Available awinda tags in clean_data (preview): {awinda_preview}\n"
            f"Missing entries:{details}"
        )

    _clear_runtime_top_level(
        runtime_train_insole_dir,
        runtime_train_skel_dir,
        runtime_test_insole_dir,
        runtime_test_skel_dir,
    )

    train_routing = _route_split(
        train_files,
        "training",
        raw_awinda_dir,
        raw_insole_dir,
        runtime_train_skel_dir,
        runtime_train_insole_dir,
        runtime_test_skel_dir,
        runtime_test_insole_dir,
    )
    test_routing = _route_split(
        test_files,
        "test",
        raw_awinda_dir,
        raw_insole_dir,
        runtime_train_skel_dir,
        runtime_train_insole_dir,
        runtime_test_skel_dir,
        runtime_test_insole_dir,
    )

    _validate_runtime_exact_keys(
        runtime_train_insole_dir,
        runtime_train_skel_dir,
        runtime_test_insole_dir,
        runtime_test_skel_dir,
        train_files,
        test_files,
    )

    print("Runtime routing completed from clean_data using train_files/test_files.")
    print(f"  clean_data source: {cache_root}")
    print(f"  Training pairs routed: {len(train_routing)}")
    print(f"  Test pairs routed: {len(test_routing)}")

    if train_routing:
        print("\nTraining routes:")
        for row in train_routing:
            print(f"  - {row['key']} -> {row['awinda_dst']} | {row['insole_dst']}")

    if test_routing:
        print("\nTest routes:")
        for row in test_routing:
            print(f"  - {row['key']} -> {row['awinda_dst']} | {row['insole_dst']}")

    return {
        "cache_root": cache_root,
        "train_routing": train_routing,
        "test_routing": test_routing,
    }


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
        "--pose_loss_mode",
        "--use_time_feature",
        "--use_gradient_data",
        "--grad_window_length",
        "--grad_polyorder",
        "--grad_smooth_grad1",
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
