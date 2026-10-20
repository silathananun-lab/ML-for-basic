import os
import cv2
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Select the model wigh parth 
    best_model_path = os.path.join(base_dir, "runs", "detect", "exp_20_epochs", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(base_dir, "runs", "detect", "exp_10_epochs", "weights", "best.pt")
        if not os.path.exists(best_model_path):
            best_model_path = "yolov8n.pt"

    print(f"Loading model for Real-time test: {best_model_path}")
    model = YOLO(best_model_path)

    # Open the Webcam (Index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit Real-time detection.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # detect the opject in the each frame 
        results = model.predict(frame, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()

        # Display the Real-time
        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()