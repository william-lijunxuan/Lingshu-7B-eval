import json
import shutil
from pathlib import Path


JSONL_PATH = Path("/root/dataset/skin/Derm1M/eval_Derm1M_train_json_1k.jsonl")
SRC_ROOT = JSONL_PATH.parent  # /root/dataset/skin/Derm1M
DST_DIR = SRC_ROOT / "clean"


def first_level_dir(rel_path: str) -> str:
    p = Path(rel_path.replace("\\", "/").lstrip("/"))
    parts = p.parts
    if len(parts) >= 2:
        return parts[0]
    return ""


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    missing = 0
    errors = 0
    skipped_exists = 0

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            rel = obj.get("image", "")
            if not rel:
                errors += 1
                continue

            rel_norm = rel.replace("\\", "/").lstrip("/")
            src = SRC_ROOT / rel_norm
            if not src.exists():
                missing += 1
                continue

            top_dir = first_level_dir(rel_norm)
            filename = Path(rel_norm).name

            dst_subdir = DST_DIR / top_dir if top_dir else DST_DIR
            dst_subdir.mkdir(parents=True, exist_ok=True)

            dst = dst_subdir / filename
            if dst.exists():
                skipped_exists += 1
                continue

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                errors += 1

    print(
        f"Done. total_lines={total}, copied={copied}, missing={missing}, "
        f"errors={errors}, skipped_exists={skipped_exists}, dst='{DST_DIR}'"
    )


if __name__ == "__main__":
    main()