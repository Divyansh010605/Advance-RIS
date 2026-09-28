import os
import sys
import json
import argparse
from pathlib import Path

# Add root of ai-service and current directory to sys.path so modules are found regardless of how python is invoked
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

cwd = Path.cwd()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from datasets.unified_dataset import parse_tbx11k_dataset

def main():
    parser = argparse.ArgumentParser(description="Parse TBX11K raw dataset into train/val/test JSON splits.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw TBX11K directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to save processed JSON lists")
    args = parser.parse_args()

    if args.raw_dir is None:
        archive_path = AI_SERVICE_DIR / "datasets" / "archive" / "TBX11K"
        if archive_path.exists():
            raw_dir = archive_path
        else:
            raw_dir = Path("data/raw/tbx11k")
    else:
        raw_dir = Path(args.raw_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        data_list = parse_tbx11k_dataset(str(raw_dir), split=split)
        out_path = output_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(data_list, f, indent=2)
        print(f"[{split.upper()}] Processed {len(data_list)} samples -> saved to {out_path}")

if __name__ == "__main__":
    main()
