import cv2
from ultralytics import YOLO

# Load a lightweight, pre-trained YOLO model (auto-downloads on first run)
# Note: Adjust the model path if your assignment specifically provided "YOLO26" weights
model = YOLO('yolov8n.pt') 

# Load your test image (a screenshot of a virtual meeting)
image_path = 'meeting_grid.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Run YOLO inference on the frame
    results = model(frame)

    attendance_count = 0

    # Process the results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class 0 in the COCO dataset is 'person'
            if int(box.cls[0]) == 0: 
                attendance_count += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw a green bounding box around the attendee
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add a label
                cv2.putText(frame, 'Attendee', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the final attendance count clearly on the top left of the image
    cv2.putText(frame, f'Total Attendance: {attendance_count}', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show the final processed image
    cv2.imshow('Automated Meeting Attendance', frame)
    
    print(f"Demo Complete: {attendance_count} attendees logged.")
    
    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()