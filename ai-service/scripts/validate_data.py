import os
import json
import argparse
from collections import Counter
from pathlib import Path

def validate_data_list(data_list_path: str, classes: list):
    """
    Validates a JSON data list against the classes and checks for missing files.
    """
    if not os.path.exists(data_list_path):
        print(f"File not found: {data_list_path}")
        return

    with open(data_list_path, 'r') as f:
        data = json.load(f)
        
    total_images = len(data)
    missing_images = 0
    class_distribution = Counter()
    invalid_labels = set()
    
    for item in data:
        img_path = item.get("path")
        labels = item.get("labels", [])
        
        if not img_path or not os.path.exists(img_path):
            missing_images += 1
            
        for label in labels:
            if label in classes:
                class_distribution[label] += 1
            else:
                invalid_labels.add(label)
                
    report = {
        "total_images": total_images,
        "missing_images": missing_images,
        "class_distribution": dict(class_distribution),
        "invalid_labels_found": list(invalid_labels)
    }
    
    print("\n--- Validation Report ---")
    print(json.dumps(report, indent=4))
    
    # Save report
    report_path = Path(data_list_path).parent / "data_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", required=True, help="Path to data list JSON")
    parser.add_argument("--classes_json", required=True, help="Path to classes.json config")
    args = parser.parse_args()
    
    with open(args.classes_json, 'r') as f:
        classes_config = json.load(f)
        classes = classes_config.get("classes", [])
        
    validate_data_list(args.data_json, classes)
