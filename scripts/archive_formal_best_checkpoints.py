#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Archive formal training folders while keeping only best.p under models/."
    )
    parser.add_argument("--src_root", default="tmp", help="Root directory to search, default: tmp")
    parser.add_argument("--dst_root", default="checkpoints", help="Archive root, default: checkpoints")
    parser.add_argument("--pattern", default="formal-*", help="Training folder name pattern")
    parser.add_argument("--expected_count", type=int, default=5, help="Expected formal folders; use 0 to disable")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing archived folders")
    parser.add_argument("--move-source", action="store_true", help="Remove source folders after a successful copy")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without copying")
    return parser.parse_args()


def has_best_checkpoint(run_dir):
    return any(path.name == "best.p" for path in (run_dir / "models").rglob("best.p"))


def discover_runs(src_root, pattern):
    runs = []
    for path in sorted(src_root.rglob(pattern)):
        if path.is_dir() and (path / "models").is_dir() and has_best_checkpoint(path):
            runs.append(path)
    return runs


def under_models(rel_path):
    return "models" in rel_path.parts


def should_copy_file(rel_path):
    if not under_models(rel_path):
        return True
    return rel_path.name == "best.p"


def copy_best_archive(src_dir, dst_dir):
    copied = []
    skipped_model_files = 0

    for src_path in src_dir.rglob("*"):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        if not should_copy_file(rel_path):
            skipped_model_files += 1
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)

    best_files = [path for path in copied if path.name == "best.p" and "models" in path.parts]
    if not best_files:
        raise RuntimeError("No best.p was copied for %s" % src_dir)

    bad_model_files = [
        path for path in dst_dir.rglob("*")
        if path.is_file() and "models" in path.parts and path.name != "best.p"
    ]
    if bad_model_files:
        raise RuntimeError("Non-best model files remained in %s: %s" % (dst_dir, bad_model_files[:5]))

    return len(copied), len(best_files), skipped_model_files


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError("Source root does not exist: %s" % src_root)

    runs = discover_runs(src_root, args.pattern)
    if args.expected_count and len(runs) != args.expected_count:
        raise RuntimeError(
            "Expected %d run folders, found %d: %s"
            % (args.expected_count, len(runs), [str(path) for path in runs])
        )

    print("Found %d formal run folders:" % len(runs))
    for run in runs:
        rel = run.relative_to(src_root)
        print("  %s -> %s" % (run, dst_root / rel))

    if args.dry_run:
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    for run in runs:
        rel = run.relative_to(src_root)
        dst_dir = dst_root / rel

        if dst_dir.exists():
            if not args.overwrite:
                raise FileExistsError("Destination exists, use --overwrite: %s" % dst_dir)
            shutil.rmtree(dst_dir)

        copied, best_files, skipped = copy_best_archive(run, dst_dir)
        print(
            "Archived %s -> %s | copied=%d best_p=%d skipped_model_files=%d"
            % (run, dst_dir, copied, best_files, skipped)
        )

        if args.move_source:
            shutil.rmtree(run)
            print("Removed source: %s" % run)


if __name__ == "__main__":
    main()
