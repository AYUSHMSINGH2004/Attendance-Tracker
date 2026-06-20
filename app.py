import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("📌 AI Attendance Tracker")
st.write("YOLOv8 + OpenCV based automatic attendance detection")


uploaded_file = st.file_uploader(
    "Upload Meeting Screenshot",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    img = np.array(image)

    st.subheader("Uploaded Image")
    st.image(img)


    # YOLO Detection
    results = model(img)


    count = 0


    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:

                count += 1

                x1,y1,x2,y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    img,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )


    st.subheader("Detection Result")

    st.image(img)

    st.success(
        f"Total Attendance Detected: {count}"
    )
