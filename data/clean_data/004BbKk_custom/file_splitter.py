"""
Truncate training and test files in this folder:
  - _training.csv / _training.txt  →  keep first 10 000 rows (header excluded)
  - _test.csv     / _test.txt      →  keep last  2 000 rows (header excluded)

Files whose name contains "1_tech_vide" or "2_tech_paos" are skipped
(already processed).
"""

from pathlib import Path

SKIP_PATTERNS = None
TRAINING_KEEP = 8_000
TEST_KEEP = 1_600


def _should_skip(name: str) -> bool:
    return False if SKIP_PATTERNS is None else any(pat in name for pat in SKIP_PATTERNS)


def _truncate(path: Path, keep: int, from_end: bool) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    if not lines:
        print(f"  SKIP (empty): {path.name}")
        return

    header, data = lines[0], lines[1:]

    if from_end:
        kept = data[-keep:] if len(data) > keep else data
        action = "last"
    else:
        kept = data[:keep]
        action = "first"

    if len(data) <= keep:
        print(f"  OK (already <= {keep} data rows): {path.name}")
        return

    path.write_text(header + "".join(kept), encoding="utf-8")
    print(f"  DONE ({action} {keep} of {len(data)} data rows): {path.name}")


def main() -> None:
    folder = Path(__file__).parent

    for path in sorted(folder.iterdir()):
        name = path.name

        if _should_skip(name):
            print(f"  SKIP (excluded pattern): {name}")
            continue

        if name.endswith("_training.csv") or name.endswith("_training.txt"):
            _truncate(path, TRAINING_KEEP, from_end=False)
        elif name.endswith("_test.csv") or name.endswith("_test.txt"):
            _truncate(path, TEST_KEEP, from_end=True)


if __name__ == "__main__":
    main()
