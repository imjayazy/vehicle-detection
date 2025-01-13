
import cv2
import os
from ultralytics import YOLO
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


image_folder = r"C:\Users\jayav\OneDrive\Desktop\infosys\pic" 
output_folder = r"C:\Users\jayav\OneDrive\Desktop\infosys\output_images"

target_size = (600, 400)

# Load two models
vehicle_model = YOLO(r"C:\Users\jayav\OneDrive\Desktop\infosys\yolov8n.pt")  
license_plate_model = YOLO(r"C:\Users\jayav\OneDrive\Desktop\infosys\best (4).pt")  


os.makedirs(output_folder, exist_ok=True)


for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
        file_path = os.path.join(image_folder, file_name)

        image = cv2.imread(file_path)

        # Resize the image to the target size
        image_resized = cv2.resize(image, target_size)

        # Overlay the image filename on the image
        cv2.putText(image_resized, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text

       
        vehicle_results = vehicle_model(image_resized)

        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                class_id = int(box.cls[0])  
                confidence = box.conf[0]  
                vehicle_label = vehicle_model.names[class_id]  

                
                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                
                label_text = f"{vehicle_label} ({confidence:.2f})"
                cv2.putText(image_resized, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

        # Detect license plates
        license_plate_results = license_plate_model(image_resized)

        for result in license_plate_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  

                
                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)

                
                roi = image_resized[y1:y2, x1:x2]

                # Use Tesseract OCR to extract text
                license_plate_text = pytesseract.image_to_string(roi, config="--psm 7").strip()

                
                print(f"Detected License Plate: {license_plate_text}")

        
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, image_resized)

        
        cv2.imshow('Vehicle and License Plate Detection', image_resized)
        if cv2.waitKey(0) & 0xFF == ord('q'):  
            break

cv2.destroyAllWindows()

print(f"Processed images are saved in: {output_folder}")
