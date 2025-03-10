import cv2
import torch
from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
model = YOLO("/Users/my/Desktop/FYP2/custom_yolo_finetuned2/weights/best.pt")  # Change "best.pt" to your trained model's file path

# Define the folder to save "without helmet" detections
save_folder = "without_helmet_detections"
os.makedirs(save_folder, exist_ok=True)

# Define class names (update according to your model's labels)
class_names = ["bus", "car", "jeepney", "motorcycle", "tricycle", "truck", "van", "with helmet", "without helmet"]

# Function to process an image
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    save_flag = False  # Flag to check if "without helmet" is detected

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{class_names[cls]} {conf:.2f}"

            # Draw bounding box
            color = (0, 255, 0) if class_names[cls] == "with helmet" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the image if "without helmet" is detected
            if class_names[cls] == "without helmet":
                save_flag = True

    # Save processed image
    processed_image_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(processed_image_path, image)

    # Save to "without_helmet_detections" if detected
    if save_flag:
        save_path = os.path.join(save_folder, os.path.basename(image_path))
        cv2.imwrite(save_path, image)

    print(f"Processed image saved: {processed_image_path}")
    if save_flag:
        print(f"'Without helmet' detected, image saved to {save_path}")

# Example usage
image_path = "/Users/my/Desktop/FYP2/testimages/without3.jpg"  # Replace with your test image path
process_image(image_path)
