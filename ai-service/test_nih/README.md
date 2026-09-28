# NIH ChestX-ray14 training

This project trains a multi-label DenseNet-121 on the 14 labeled findings in the NIH ChestX-ray14 dataset and saves the best validation checkpoint.

## Dataset layout

Download and extract the NIH dataset, then point the command at:

- `Data_Entry_2017.csv`
- the image root containing the images (the loader also searches nested folders such as `images_001/images`)

Install PyTorch, torchvision, and Pillow in the Python environment used to run the scripts.

## Train

```powershell
python train_nih.py --csv_path "data/nih/Data_Entry_2017.csv" --images_dir "data/nih" --epochs 5 --batch_size 16
```

The default paths are relative to this project. The model is saved to `saved_models/nih_densenet121.pth`; validation uses a patient-level split to avoid putting images from one patient into both sets. Add `--pretrained` to initialize DenseNet-121 from ImageNet weights (requires downloading torchvision weights).

Useful options: `--save_dir`, `--val_fraction`, `--workers`, and `--seed`. Training requires the real CSV and images and stops with a clear error if they are missing.

## Predict

```powershell
python predict_nih.py --image "path/to/xray.png" --model_path "saved_models/nih_densenet121.pth"
```

The checkpoint includes the model weights and ordered class names.
