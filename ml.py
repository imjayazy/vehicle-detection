import cv2
import os
import torch
from ultralytics import YOLO

# Correct folder paths


image_folder = r"C:\Users\jayav\OneDrive\Desktop\infosys\pic"  # Update to your folder path
output_folder = r"C:\Users\jayav\OneDrive\Desktop\infosys\output_images"

# Desired output size for the images (width, height)
target_size = (600, 400)  # Update with the desired width and height

# Load the YOLO model()
model = YOLO("yolov8n.pt")  # Use a pre-trained YOLOv8 model

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the folder
for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
        file_path = os.path.join(image_folder, file_name)

        # Load the image
        image = cv2.imread(file_path)

        # Resize the image to the target size
        image_resized = cv2.resize(image, target_size)

        # Perform object detection
        results = model(image_resized)

        # Draw bounding boxes and labels on the image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                confidence = box.conf[0]  # Get confidence score
                label = result.names[int(box.cls[0])]  # Get class name

                # Draw the bounding box
                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw the label and confidence
                cv2.putText(image_resized, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image_resized, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)  # Red text color            

        # Save the processed image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, image_resized)

        # Display the processed image with a delay of 2ms
        cv2.imshow('Image with Detection', image_resized)
        if cv2.waitKey(2000) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

cv2.destroyAllWindows()

print(f"Processed images are saved in: {output_folder}")
