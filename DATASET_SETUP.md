# NIH ChestX-ray14 Dataset Setup & Download Guide

This guide details the setup, directory layout, and preparation steps for the  
**NIH ChestX-ray14** dataset used for training and evaluating the 14-class DenseNet-121 model.

---

## 1. Dataset Overview: NIH ChestX-ray14

The **NIH ChestX-ray14** dataset contains **112,120 frontal-view chest X-ray images**  
from 30,805 unique patients, each annotated with up to 14 thoracic pathology labels  
(NLP-mined from radiology reports).

### Target Classes (14 findings)

| Label              | Description                                 |
|--------------------|---------------------------------------------|
| Atelectasis        | Partial or complete lung collapse           |
| Cardiomegaly       | Enlarged heart                              |
| Effusion           | Fluid in the pleural space                  |
| Infiltration       | Pulmonary infiltrate                        |
| Mass               | Soft-tissue density ≥ 3 cm                  |
| Nodule             | Rounded opacity < 3 cm                      |
| Pneumonia          | Lung infection / consolidation              |
| Pneumothorax       | Air in the pleural space                    |
| Consolidation      | Airspace consolidation                      |
| Edema              | Pulmonary oedema                            |
| Emphysema          | Overinflated air sacs                       |
| Fibrosis           | Lung scarring / interstitial fibrosis       |
| Pleural_Thickening | Pleural thickening                          |
| Hernia             | Diaphragmatic hernia                        |

Labels are **multi-hot** (a single image can carry multiple labels).  
The `No Finding` label is excluded during training.

---

## 2. Official Download

Download the dataset from the **NIH Clinical Center**:

- **NIH Official Page**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Kaggle Mirror**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Academic Torrents**: search "NIH ChestX-ray14"

Files to download:

| File                          | Size    | Notes                                      |
|-------------------------------|---------|--------------------------------------------|
| `images_001.tar.gz` … `images_012.tar.gz` | ~42 GB total | 12 image archive parts       |
| `Data_Entry_2017_v2020.csv`   | ~7.9 MB | Label metadata (one row per image)         |
| `train_val_list.txt`          | optional | Official train/val split                  |
| `test_list.txt`               | optional | Official test split                       |

---

## 3. Directory Layout

Place the extracted files under `ai-service/test_nih/` as shown:

```
ai-service/
└── test_nih/
    ├── Data_Entry_2017_v2020.csv      ← label metadata (git-ignored)
    ├── nih_classes.json
    ├── nih_dataset.py
    ├── train_nih.py
    ├── predict_nih.py
    ├── images/                        ← extracted image archives (git-ignored)
    │   ├── 00000001_000.png
    │   ├── 00000001_001.png
    │   └── …  (112 120 images total)
    └── saved_models/
        └── nih_densenet121.pth        ← trained checkpoint (git-ignored, ~85 MB)
```

> **Nested layout supported**: The dataset loader (`nih_dataset.py`) recursively  
> searches for images, so `images_001/images/`, `images_002/images/`, … also work.

---

## 4. Extracting Image Archives

```bash
# Extract all 12 parts into ai-service/test_nih/images/
cd ai-service/test_nih
mkdir -p images
for i in $(seq -w 001 012); do
    tar -xzf images_${i}.tar.gz -C images --strip-components=1
done
```

On Windows (PowerShell):

```powershell
cd ai-service\test_nih
New-Item -ItemType Directory -Force images
1..12 | ForEach-Object {
    $part = "images_{0:D3}.tar.gz" -f $_
    tar -xzf $part -C images --strip-components=1
}
```

---

## 5. Training the Model

```bash
cd ai-service/test_nih

# Basic (random init, 5 epochs)
python train_nih.py \
    --csv_path Data_Entry_2017_v2020.csv \
    --images_dir images \
    --epochs 5 \
    --batch_size 16

# Recommended (ImageNet pretrained init, 20 epochs)
python train_nih.py \
    --csv_path Data_Entry_2017_v2020.csv \
    --images_dir images \
    --epochs 20 \
    --batch_size 32 \
    --pretrained
```

The best validation checkpoint is saved automatically to:  
`ai-service/test_nih/saved_models/nih_densenet121.pth`

### Training Options

| Flag              | Default      | Description                                 |
|-------------------|--------------|---------------------------------------------|
| `--csv_path`      | auto-detect  | Path to `Data_Entry_2017_v2020.csv`         |
| `--images_dir`    | `images/`    | Root directory containing the PNG images    |
| `--epochs`        | 5            | Number of training epochs                   |
| `--batch_size`    | 16           | Images per batch                            |
| `--lr`            | 1e-4         | Learning rate (AdamW)                       |
| `--val_fraction`  | 0.1          | Fraction of patients held out for validation|
| `--workers`       | 0            | DataLoader workers (0 = safe on Windows)    |
| `--pretrained`    | off          | Use ImageNet-pretrained DenseNet weights    |
| `--save_dir`      | `saved_models/` | Checkpoint output directory             |
| `--seed`          | 42           | Random seed for reproducibility             |

> Validation is done on a **patient-level split** so no patient appears in both  
> train and validation sets — this avoids data leakage.

---

## 6. Running Inference (CLI)

```bash
cd ai-service/test_nih
python predict_nih.py \
    --image /path/to/xray.png \
    --model_path saved_models/nih_densenet121.pth
```

Sample output:

```
--- NIH 14-Class Predictions for 00000001_000.png ---
  [Positive] Atelectasis        : 67.34%
  [Negative] Cardiomegaly       : 12.10%
  [Negative] Effusion           : 28.90%
  ...
```

---

## 7. Docker Volume Mount

The trained checkpoint is **not baked into the Docker image** (too large for CI/CD).  
It is mounted at container startup via the `docker-compose.yml` volume:

```yaml
volumes:
  - ${MODEL_HOST_PATH:-./ai-service/test_nih/saved_models}:/app/test_nih/saved_models
```

Set `MODEL_HOST_PATH` in your `.env` file if the checkpoint lives elsewhere:

```env
MODEL_HOST_PATH=/data/models/nih
```

Or specify it inline:

```bash
MODEL_HOST_PATH=/data/models/nih docker compose up -d
```
