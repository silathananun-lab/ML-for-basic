import os
import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Select the beat model weigh from train  (If training has not yet been performed, yolov8n.pt will be used as a fallback.)
    best_model_path = os.path.join(base_dir, "runs", "detect", "exp_20_epochs", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(base_dir, "runs", "detect", "exp_10_epochs", "weights", "best.pt")
        if not os.path.exists(best_model_path):
            best_model_path = "yolov8n.pt"

    print(f"Loading model from: {best_model_path}")
    model = YOLO(best_model_path)

    # Pull images fill from dataset/test/images
    test_img_dir = os.path.join(base_dir, "dataset", "test", "images")
    image_paths = glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png"))

    if not image_paths:
        print("No test images found in dataset/test/images!")
        return

    sample_image_path = image_paths[0]
    print(f"Testing on image: {sample_image_path}")

    # detect and classify objects
    results = model.predict(source=sample_image_path, conf=0.25)

    # Build the Bounding Box, Label, Confidence Score
    res_plotted = results[0].plot()

    # Display Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.title("Object Detection Result - YOLO")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()