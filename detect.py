import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load the YOLO model
model = YOLO("/home/hydro/Documents/yolov11-20250218T062338Z-001/yolov11/runs/detect/train/weights/best.pt")  # segmentation model
names = model.model.names

# Video input and output setup (using camera)
input_source = "/home/hydro/Downloads/okecoba.mp4" 
output_dest = "/home/hydro/Documents/output_file"  # Output video file path

# Open the camera
cap = cv2.VideoCapture(input_source)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Set the resolution of the camera to a wider aspect ratio (e.g., 1280x720 or 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Get video properties after setting new resolution
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_dest, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break

    # Perform YOLO inference on the full frame
    results = model.predict(frame, conf=0.4, iou=0.7, retina_masks=True)
    annotated_frame = frame.copy()

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.data.cpu().numpy()
        scores = results[0].boxes.conf.cpu().tolist()

        for mask, cls, score in zip(masks, clss, scores):
            mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors(int(cls), True)
            text_label = f"{names[int(cls)]} {score:.2f}"

            for contour in contours:
                cv2.drawContours(annotated_frame, [contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(annotated_frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write the annotated frame to the output file
    out.write(annotated_frame)
    annotated_frame = cv2.resize(annotated_frame, (1920, 1080))
    
    # Display the annotated frame
    cv2.imshow("YOLO Inference", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
