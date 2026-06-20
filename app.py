import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os


# Page config
st.set_page_config(
    page_title="AI Attendance Tracker",
    page_icon="📌"
)


# Load YOLO model once
@st.cache_resource
def load_model():

    model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        st.error("YOLO model file not found")
        st.stop()

    return YOLO(model_path)



model = load_model()



st.title("📌 AI Attendance Tracker")

st.write(
    "YOLOv8 + OpenCV based automated attendance detection"
)



uploaded_file = st.file_uploader(
    "Upload Meeting Screenshot",
    type=["jpg","jpeg","png"]
)



if uploaded_file:


    image = Image.open(uploaded_file)

    img = np.array(image)


    st.subheader("Original Image")

    st.image(img)



    with st.spinner("Detecting attendees..."):


        results = model(img)



        attendance = 0



        for result in results:


            for box in result.boxes:


                cls = int(box.cls[0])


                # Person class
                if cls == 0:


                    attendance += 1


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


                    cv2.putText(
                        img,
                        "Person",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2
                    )



    st.subheader("Detection Result")


    st.image(img)


    st.success(
        f"Total Attendance Detected: {attendance}"
    )
