import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def parse_nih_csv(csv_path: str, images_dir: str) -> List[Dict[str, Any]]:
    """
    Parses NIH Data_Entry_2017.csv / Data_Entry_2017_v2020.csv into records.
    Builds a filename index to support flat or nested image directories.
    """
    csv_file = Path(csv_path)
    image_root = Path(images_dir)
    if not csv_file.is_file():
        raise FileNotFoundError(f"NIH CSV file not found: {csv_file}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"NIH image directory not found: {image_root}")

    # Build index of image filenames to paths
    image_index: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in image_root.rglob(ext):
            image_index[p.name] = p

    parsed_records = []
    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"Image Index", "Finding Labels", "Patient ID"}
        if not reader.fieldnames or not required_cols.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required_cols))}")

        for row in reader:
            img_name = row.get("Image Index", "").strip()
            findings_str = row.get("Finding Labels", "").strip()
            patient_id = row.get("Patient ID", "").strip()

            if not img_name:
                continue

            img_path = image_index.get(img_name)
            if img_path is None:
                direct_path = image_root / img_name
                if direct_path.is_file():
                    img_path = direct_path
                else:
                    continue

            labels = [
                lbl.strip().replace(" ", "_")
                for lbl in findings_str.split("|")
                if lbl.strip() and lbl.strip() != "No Finding"
            ]

            parsed_records.append({
                "image_name": img_name,
                "path": str(img_path),
                "labels": labels,
                "patient_id": patient_id,
                "view_position": row.get("View Position", "").strip(),
                "patient_age": row.get("Patient Age", "").strip(),
                "patient_gender": row.get("Patient Gender", "").strip(),
            })

    if not parsed_records:
        raise ValueError(f"No matching images found under {image_root} for {csv_file}")
    logger.info("Parsed %d valid image records from %s", len(parsed_records), csv_path)
    return parsed_records


def assert_no_patient_leakage(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Verifies that no patient ID appears in more than one split."""
    train_patients = {r["patient_id"] for r in train_rows}
    val_patients = {r["patient_id"] for r in val_rows}

    train_val_leak = train_patients.intersection(val_patients)
    assert not train_val_leak, f"Patient leakage between train and val: {len(train_val_leak)} patients"

    if test_rows:
        test_patients = {r["patient_id"] for r in test_rows}
        train_test_leak = train_patients.intersection(test_patients)
        assert not train_test_leak, f"Patient leakage between train and test: {len(train_test_leak)} patients"

        val_test_leak = val_patients.intersection(test_patients)
        assert not val_test_leak, f"Patient leakage between val and test: {len(val_test_leak)} patients"


def _read_filename_list(list_path: Path) -> Set[str]:
    with list_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def create_patient_splits(
    records: List[Dict[str, Any]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    train_val_list_path: Optional[str] = None,
    test_list_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions dataset by patient ID. Uses official train_val_list / test_list if provided,
    otherwise performs random patient-level partitioning.
    """
    rng = random.Random(seed)

    # 1. Check if official split files are provided and exist
    use_official = False
    if train_val_list_path and test_list_path:
        tv_file, test_file = Path(train_val_list_path), Path(test_list_path)
        if tv_file.is_file() and test_file.is_file():
            use_official = True
            tv_names = _read_filename_list(tv_file)
            test_names = _read_filename_list(test_file)

            tv_records = [r for r in records if r["image_name"] in tv_names]
            test_records = [r for r in records if r["image_name"] in test_names]
            logger.info("Using official NIH split lists: %d train_val, %d test", len(tv_records), len(test_records))
        else:
            logger.warning("Official split files specified but not found. Falling back to patient-level split.")

    if not use_official:
        # Group records by patient ID
        patients = sorted({r["patient_id"] for r in records})
        rng.shuffle(patients)

        if test_fraction > 0.0:
            num_test = max(1, round(len(patients) * test_fraction))
            test_patients = set(patients[:num_test])
            tv_patients = patients[num_test:]
        else:
            test_patients = set()
            tv_patients = patients

        test_records = [r for r in records if r["patient_id"] in test_patients]
        tv_records = [r for r in records if r["patient_id"] not in test_patients]

    # 2. Split tv_records into train and validation by patient ID
    tv_patient_ids = sorted({r["patient_id"] for r in tv_records})
    rng.shuffle(tv_patient_ids)

    num_val = max(1, round(len(tv_patient_ids) * val_fraction)) if len(tv_patient_ids) > 1 else 0
    val_patients = set(tv_patient_ids[:num_val])

    train_records = [r for r in tv_records if r["patient_id"] not in val_patients]
    val_records = [r for r in tv_records if r["patient_id"] in val_patients]

    assert_no_patient_leakage(train_records, val_records, test_records if test_records else None)

    logger.info(
        "Split complete: %d train images, %d val images, %d test images (0 leakage)",
        len(train_records),
        len(val_records),
        len(test_records),
    )
    return train_records, val_records, test_records
