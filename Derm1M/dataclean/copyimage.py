import json
import shutil
from pathlib import Path


# JSONL_PATH = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean.jsonl")
JSONL_PATH = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean.jsonl")
SRC_ROOT = JSONL_PATH.parent  # /home/william/dataset/skin/Derm1M
DST_DIR = SRC_ROOT / "1kclean"


def safe_dst_name(rel_path: str) -> str:
    # Avoid collisions by encoding subfolders into filename
    # e.g. "youtube/a/b.jpg" -> "youtube__a__b.jpg"
    return rel_path.replace("\\", "/").strip("/").replace("/", "__")


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    missing = 0
    errors = 0

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

            src = SRC_ROOT / rel
            if not src.exists():
                missing += 1
                continue

            dst = DST_DIR / safe_dst_name(rel)
            if dst.exists():
                copied += 1
                continue

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                errors += 1

    print(
        f"Done. total_lines={total}, copied={copied}, missing={missing}, errors={errors}, dst='{DST_DIR}'"
    )


if __name__ == "__main__":
    main()
