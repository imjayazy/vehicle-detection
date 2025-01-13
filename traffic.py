import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model for vehicle detection
vehicle_model = YOLO(r"C:\Users\jayav\OneDrive\Desktop\infosys\yolov8n.pt")  # Replace with your YOLO model path

# Function to draw a traffic signal in the corner
def draw_traffic_signal(frame, position, is_green):
    # Define the position for the signal (top-left corner)
    x, y = position
    radius = 25  # Radius of the signal light

    # Set color based on traffic light state
    if is_green:
        signal_color = (0, 255, 0)  # Green light
    else:
        signal_color = (0, 0, 255)  # Red light
    
    # Draw the traffic light (a green or red light)
    cv2.circle(frame, (x, y), radius, signal_color, -1)  # Circle representing the signal
    return frame

# Function to detect vehicles and return the density
def get_vehicle_density(frame):
    # Perform vehicle detection using YOLO
    results = vehicle_model(frame)
    
    # Count the number of detected vehicles
    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if label == 'car' or label == 'truck' or label == 'bus':  # Check for vehicle types
                vehicle_count += 1
    return vehicle_count

# Open video streams
cap1 = cv2.VideoCapture('video1.mp4')
cap2 = cv2.VideoCapture('video2.mp4')
cap3 = cv2.VideoCapture('video3.mp4')
cap4 = cv2.VideoCapture('video4.mp4')

while True:
    # Read frames from the video streams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # If any frame is not available, break the loop
    if not ret1 or not ret2 or not ret3 or not ret4:
        break

    # Resize frames if necessary (for consistency in display size)
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    frame3 = cv2.resize(frame3, (640, 480))
    frame4 = cv2.resize(frame4, (640, 480))

    # Detect vehicles and get the density for each frame
    density1 = get_vehicle_density(frame1)
    density2 = get_vehicle_density(frame2)
    density3 = get_vehicle_density(frame3)
    density4 = get_vehicle_density(frame4)

    # Determine which frame has the highest vehicle density
    densities = [density1, density2, density3, density4]
    max_density_index = densities.index(max(densities))

    # Set the traffic light to green for the frame with the highest density
    is_green1 = max_density_index == 0
    is_green2 = max_density_index == 1
    is_green3 = max_density_index == 2
    is_green4 = max_density_index == 3

    # Draw traffic signals on the corners of the frames
    frame1 = draw_traffic_signal(frame1, (20, 20), is_green1)
    frame2 = draw_traffic_signal(frame2, (20, 20), is_green2)
    frame3 = draw_traffic_signal(frame3, (20, 20), is_green3)
    frame4 = draw_traffic_signal(frame4, (20, 20), is_green4)

    # Concatenate the frames to display 4 videos simultaneously in the same screen (2x2 grid)
    top_row = np.hstack((frame1, frame2))  # Combine first two videos horizontally
    bottom_row = np.hstack((frame3, frame4))  # Combine next two videos horizontally
    combined_frame = np.vstack((top_row, bottom_row))  # Stack both rows vertically

    # Display the combined frame with 4 videos playing simultaneously
    cv2.imshow('4 Videos Playing Simultaneously with Traffic Signals', combined_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
