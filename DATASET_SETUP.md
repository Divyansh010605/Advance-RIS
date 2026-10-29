# NIH ChestX-ray14 Dataset Setup & Download Guide

This guide details the setup, directory layout, and preparation steps for the  
**NIH ChestX-ray14** dataset used for training and evaluating the 14-class DenseNet-121 model.

---

## 1. Dataset Overview: NIH ChestX-ray14

The **NIH ChestX-ray14** dataset contains **112,120 frontal-view chest X-ray images**  
from 30,805 unique patients, each annotated with up to 14 thoracic pathology labels  
(NLP-mined from radiology reports).

### Target Classes (15 findings)

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
| No_Finding         | Normal chest radiograph (no acute findings) |

Labels are **multi-hot** (a single image can carry multiple labels).  
The NIH training pipeline includes `No_Finding` as its 15th label. In production XRV inference, `No_Finding` is derived from the 14 pathology scores because the selected TorchXRayVision weights do not include a trained `No Finding` output.

---

## 2. Official Download

Download the dataset from the **NIH Clinical Center**:

- **NIH Official Page**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Kaggle Mirror**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Academic Torrents**: search "NIH ChestX-ray14"

Files to download:

| File                          | Size    | Notes                                      |
|-------------------------------|---------|--------------------------------------------|
| `images_001.tar.gz` … `images_012.tar.gz` | ~42 GB total | 12 image archive parts    |
| `Data_Entry_2017_v2020.csv`   | ~7.9 MB | Label metadata (one row per image)         |
| `train_val_list.txt`          | optional | Official train/val split                  |
| `test_list.txt`               | optional | Official test split                       |

---

## 3. Directory Layout

Place the extracted files under `ai-service/test_nih/` as shown:

```
ai-service/
└── test_nih/
    ├── Data_Entry_2017_v2020.csv      ← label metadata (7.8 MB)
    ├── images/                        ← extracted image archives (PNG images)
    │   ├── 00000001_000.png
    │   └── …  (112 120 images total)
    └── saved_models/                  # Optional local research checkpoints
```

> **Nested layout supported**: The dataset loader (`src/data/dataset.py`) recursively  
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

## 5. Training the Models (Config-Driven Retraining Pipeline)

Retraining uses the unified `src.train` module and YAML configurations:

```bash
cd ai-service

# Retrain DenseNet-121
python -m src.train --config configs/densenet121.yaml

# Retrain ResNet-50
python -m src.train --config configs/resnet50.yaml

# Retrain Swin Transformer Tiny
python -m src.train --config configs/swin_tiny.yaml

# Retrain all three sequentially
bash scripts/retrain_all.sh       # Linux/macOS
.\scripts\retrain_all.ps1        # Windows
```

Checkpoints and metrics (AUROC, F1, sensitivity, specificity) are saved under `outputs/retrained/<model_name>/`.

---

## 6. Explainable AI (XAI) Evaluation

Evaluate model localization against NIH bounding box ground-truth (`BBox_List_2017.csv`):

```bash
cd ai-service
python -m src.eval.xai_eval \
    --architecture densenet121 \
    --bbox_csv test_nih/BBox_List_2017.csv \
    --method gradcam++
```

---

## 7. Docker Volume Mount

Production inference does not require a locally trained NIH checkpoint. TorchXRayVision downloads `densenet121-res224-all` and `resnet50-res512-all` at first startup and caches them in its user cache directory. NIH dataset files and local checkpoints are only needed for the research training and evaluation workflows above.

```yaml
volumes:
  - ${XRV_CACHE_PATH:-./xrv-cache}:/root/.torchxrayvision
```

Set `XRV_CACHE_PATH` in your `.env` file to choose another host cache directory:

```env
XRV_CACHE_PATH=/data/cache/torchxrayvision
```

Or specify it inline:

```bash
XRV_CACHE_PATH=/data/cache/torchxrayvision docker compose up -d
```
