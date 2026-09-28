"""
DEPRECATED – This script was the original TBX11K training entry point.

The project now uses the NIH ChestX-ray14 DenseNet-121 model.
Use the training script in test_nih/ instead:

    cd ai-service/test_nih
    python train_nih.py --csv_path Data_Entry_2017_v2020.csv --images_dir images --epochs 20 --pretrained

See DATASET_SETUP.md for full instructions.
"""
import sys

if __name__ == "__main__":
    print(
        "\n[DEPRECATED] ai-service/train.py is no longer used.\n"
        "Train the NIH ChestX-ray14 model with:\n\n"
        "    cd ai-service/test_nih\n"
        "    python train_nih.py --csv_path Data_Entry_2017_v2020.csv "
        "--images_dir images --epochs 20 --pretrained\n"
    )
    sys.exit(0)
