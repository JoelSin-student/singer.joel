from pathlib import Path
import sys
from datetime import datetime
import re
import hashlib

from sources.util import run_cmd_streaming


# =========================
# User controls
# =========================
MODEL = "transformer_encoder"
CV_CONFIG = "sources/config/transformer_encoder/cv.yaml"

# Select algorithms independently.
RUN_ORIGINAL = True
RUN_SOLEFORMER = True

# Select CV families independently.
RUN_LOSO = True
RUN_LOSITO = True
RUN_PS_LOSITO = True
RUN_PSIT_LOSO = True

# Optional selectors for per-subgroup CV families.
# Empty list = run all available groups.
PS_LOSITO_SUBJECTS = []
PSIT_LOSO_SITUATIONS = []

# Safety and execution behavior.
PRINT_ONLY = False  # True: print commands only, False: execute
STOP_ON_ERROR = True

# Output namespace safety.
AUTO_RUN_LABEL = True
RUN_LABEL = ""


# =========================
# Internals
# =========================
def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _selected_algorithms():
    values = []
    if RUN_ORIGINAL:
        values.append("original")
    if RUN_SOLEFORMER:
        values.append("soleformer")
    if not values:
        raise ValueError("At least one algorithm must be enabled (RUN_ORIGINAL / RUN_SOLEFORMER).")
    return values


def _selected_run_types():
    values = []
    if RUN_LOSO:
        values.append("LOSO")
    if RUN_LOSITO:
        values.append("LOSitO")
    if RUN_PS_LOSITO:
        values.append("PS-LOSitO")
    if RUN_PSIT_LOSO:
        values.append("PSit-LOSO")
    if not values:
        raise ValueError("At least one CV type must be enabled.")
    return values


def _campaigns():
    campaigns = []
    for algorithm in _selected_algorithms():
        for run_type in _selected_run_types():
            campaign = {
                "algorithm": algorithm,
                "run_type": run_type,
                "subjects": list(PS_LOSITO_SUBJECTS) if run_type == "PS-LOSitO" else [],
                "situations": list(PSIT_LOSO_SITUATIONS) if run_type == "PSit-LOSO" else [],
            }
            campaigns.append(campaign)
    return campaigns


def _build_cmd(root: Path, campaign: dict, run_label: str):
    cmd = [
        sys.executable,
        "-u",
        "sources/main.py",
        "cv",
        "--model",
        MODEL,
        "--config",
        CV_CONFIG,
        "--algorithms",
        campaign["algorithm"],
        "--run_types",
        campaign["run_type"],
        "--run_label",
        run_label,
    ]

    if campaign["subjects"]:
        cmd.extend(["--ps_losito_subjects", ",".join(campaign["subjects"])])

    if campaign["situations"]:
        cmd.extend(["--psit_loso_situations", ",".join(campaign["situations"])])

    return cmd


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower() if value else "na"


def _short_hash(text: str, length: int = 6) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def _campaign_suffix(campaign: dict) -> str:
    algorithm_map = {
        "original": "orig",
        "soleformer": "sole",
    }
    run_type_map = {
        "LOSO": "loso",
        "LOSitO": "losito",
        "PS-LOSitO": "psl",
        "PSit-LOSO": "psi",
    }

    parts = [algorithm_map.get(campaign["algorithm"], _slug(campaign["algorithm"]))]
    parts.append(run_type_map.get(campaign["run_type"], _slug(campaign["run_type"])))
    if campaign["run_type"] == "PS-LOSitO":
        subjects = campaign.get("subjects") or []
        if subjects:
            parts.append("subj_" + "-".join(_slug(s)[:8] for s in subjects))
    if campaign["run_type"] == "PSit-LOSO":
        situations = campaign.get("situations") or []
        if situations:
            parts.append("sit_" + "-".join(_slug(s)[:8] for s in situations))

    base = "_".join(_slug(p) for p in parts)
    return f"{base}_{_short_hash(base)}"


def _resolved_run_label(campaign: dict | None = None):
    if AUTO_RUN_LABEL:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if campaign is None:
            return f"run_{ts}"
        return f"run_{ts}_{_campaign_suffix(campaign)}"

    value = str(RUN_LABEL).strip()
    if not value:
        raise ValueError("RUN_LABEL must be non-empty when AUTO_RUN_LABEL=False")
    return value


def _estimate_trainings(campaign: dict):
    rt = campaign["run_type"]
    if rt == "LOSO":
        folds = 5
    elif rt == "LOSitO":
        folds = 5
    elif rt == "PS-LOSitO":
        if campaign["subjects"]:
            folds = 0
            for s in campaign["subjects"]:
                folds += 4 if s == "001CcSs" else 5
        else:
            folds = 24  # 4 + 5 + 5 + 5 + 5
    elif rt == "PSit-LOSO":
        if campaign["situations"]:
            folds = 0
            for sit in campaign["situations"]:
                folds += 4 if str(sit).startswith("1_") else 5
        else:
            folds = 24  # 4 + 5 + 5 + 5 + 5
    else:
        folds = 0

    return folds


def main():
    root = _repo_root()
    campaigns = _campaigns()
    global_run_label = _resolved_run_label()

    print("Planned independent CV campaigns:")
    if AUTO_RUN_LABEL:
        print("Output run_label mode: auto-rich (timestamp + algorithm + cv type + optional subgroup)")
    else:
        print(f"Output run_label mode: manual ({global_run_label})")
    total_trainings = 0
    for i, campaign in enumerate(campaigns, start=1):
        est = _estimate_trainings(campaign)
        total_trainings += est
        print(
            f"{i:02d}. algorithm={campaign['algorithm']} | run_type={campaign['run_type']} "
            f"| subjects={campaign['subjects'] or 'ALL'} | situations={campaign['situations'] or 'ALL'} "
            f"| est_trainings={est}"
        )

    print(f"Total estimated model trainings for this launch set: {total_trainings}")

    for i, campaign in enumerate(campaigns, start=1):
        run_label = _resolved_run_label(campaign) if AUTO_RUN_LABEL else global_run_label
        cmd = _build_cmd(root, campaign, run_label=run_label)
        print("-" * 80)
        print(f"[{i}/{len(campaigns)}] Running campaign")
        print(f"run_label={run_label}")
        print(" ".join(cmd))

        if PRINT_ONLY:
            continue

        try:
            run_cmd_streaming(cmd, cwd=root)
        except Exception:
            if STOP_ON_ERROR:
                raise
            print("Campaign failed but continuing because STOP_ON_ERROR=False")


if __name__ == "__main__":
    main()
