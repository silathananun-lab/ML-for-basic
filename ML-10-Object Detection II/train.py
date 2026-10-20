import os
import yaml
from ultralytics import YOLO

def main():
    # Set Relative Path go to the data.yaml
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    # adjust path in the data.yaml to match the current folder structure.
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = os.path.join(base_dir, "dataset")
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("=== Start Training Experiment 1 (10 Epochs) ===")
    model1 = YOLO("yolov8n.pt")  # Load Pretrained YOLOv8 Nano
    model1.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="exp_10_epochs"
    )

    print("\n=== Start Training Experiment 2 (20 Epochs) ===")
    model2 = YOLO("yolov8n.pt")
    model2.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="exp_20_epochs"
    )

    print("\nTraining completed! Models saved under runs/detect/")

if __name__ == "__main__":
    main()