import os
from ultralytics import YOLO

def evaluate_model(model_path, exp_name):
    print(f"\n--- Evaluating Model: {exp_name} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return

    model = YOLO(model_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    # the Validate on the Test Set or the Valid Set
    metrics = model.val(data=yaml_path, split="test")

    print(f"[{exp_name} Results]")
    print(f"  Precision (mp) : {metrics.box.mp:.4f}")
    print(f"  Recall (mr)    : {metrics.box.mr:.4f}")
    print(f"  mAP50          : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 (map) : {metrics.box.map:.4f}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # evaluate Exp 1 (10 Epochs)
    model1_path = os.path.join(base_dir, "runs", "detect", "exp_10_epochs", "weights", "best.pt")
    evaluate_model(model1_path, "10 Epochs")

    # evaluate Exp 2 (20 Epochs)
    model2_path = os.path.join(base_dir, "runs", "detect", "exp_20_epochs", "weights", "best.pt")
    evaluate_model(model2_path, "20 Epochs")

if __name__ == "__main__":
    main()