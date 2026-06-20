import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Attendance Tracker",
    page_icon="🤖",
    layout="wide"
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown(
"""
<style>


/* Main Background */

.stApp {

background:
linear-gradient(
135deg,
#0f2027,
#203a43,
#2c5364
);

color:white;

}


/* Title */

.title {

font-size:55px;
font-weight:900;
text-align:center;

background:
linear-gradient(
90deg,
#00F5A0,
#00D9F5,
#FF61D2
);

-webkit-background-clip:text;
color:transparent;

}


/* Subtitle */

.subtitle {

text-align:center;

font-size:22px;

color:#d9faff;

}


/* Cards */


.card {


background:

rgba(255,255,255,0.12);


padding:25px;

border-radius:25px;


box-shadow:

0px 10px 30px rgba(0,0,0,0.4);


backdrop-filter:

blur(15px);


margin:15px;


}



.metric {


font-size:40px;

font-weight:bold;

text-align:center;


color:#00F5A0;


}


/* Buttons */

.stButton button {


background:

linear-gradient(
90deg,
#00F5A0,
#00D9F5
);


color:black;


font-size:18px;


border-radius:20px;


padding:12px 30px;


font-weight:bold;


}



</style>

""",
unsafe_allow_html=True
)



# ---------------- MODEL ---------------- #

@st.cache_resource

def load_model():

    return YOLO("yolov8n.pt")



model = load_model()



# ---------------- HEADER ---------------- #

st.markdown(
"""
<div class="title">

🤖 AI Attendance Tracker

</div>


<div class="subtitle">

YOLOv8 + Computer Vision Powered Smart Attendance System

</div>

""",

unsafe_allow_html=True
)


st.write("")



# ---------------- SIDEBAR ---------------- #

with st.sidebar:


    st.markdown(
    """
    ## ⚙️ Control Panel

    Upload a meeting screenshot.

    The AI will:

    ✅ Detect people

    ✅ Draw bounding boxes

    ✅ Calculate attendance

    """
    )



    confidence = st.slider(
        "Detection Confidence",
        0.1,
        1.0,
        0.5
    )



# ---------------- UPLOAD ---------------- #

uploaded_file = st.file_uploader(

"📸 Upload Meeting Image",

type=[
"jpg",
"jpeg",
"png"
]

)



if uploaded_file:


    image = Image.open(uploaded_file)


    img = np.array(image)



    col1,col2 = st.columns(2)



    with col1:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.subheader(
        "📷 Original Image"
        )


        st.image(
            img,
            use_container_width=True
        )


        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )



    # -------- DETECTION -------- #


    with st.spinner(
        "🧠 AI is analyzing image..."
    ):


        time.sleep(1)


        results = model(
            img,
            conf=confidence
        )


        attendance=0



        for result in results:


            for box in result.boxes:



                cls=int(box.cls[0])



                if cls==0:



                    attendance+=1



                    x1,y1,x2,y2 = map(
                    int,
                    box.xyxy[0]
                    )



                    cv2.rectangle(

                    img,

                    (x1,y1),

                    (x2,y2),

                    (0,255,150),

                    3

                    )



                    cv2.putText(

                    img,

                    "Person",

                    (x1,y1-10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255,255,255),

                    2

                    )




    with col2:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.subheader(
        "🎯 AI Detection Result"
        )


        st.image(
            img,
            use_container_width=True
        )


        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )





# ---------------- METRICS ---------------- #


if uploaded_file:


    st.write("")


    c1,c2,c3 = st.columns(3)



    with c1:


        st.markdown(

        f"""

        <div class="card">

        <div class="metric">

        {attendance}

        </div>

        👥 Total Attendance

        </div>

        """,

        unsafe_allow_html=True

        )




    with c2:


        st.markdown(

        """

        <div class="card">

        <div class="metric">

        YOLOv8

        </div>

        🧠 AI Model

        </div>

        """,

        unsafe_allow_html=True

        )





    with c3:


        st.markdown(

        """

        <div class="card">

        <div class="metric">

        LIVE

        </div>

        ⚡ Detection

        </div>

        """,

        unsafe_allow_html=True

        )


else:


    st.markdown(

    """

    <div class="card">


    👋 Welcome!


    Upload a meeting screenshot to start AI attendance detection.


    </div>

    """,

    unsafe_allow_html=True

    )
