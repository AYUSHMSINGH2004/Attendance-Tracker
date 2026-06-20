import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI Attendance Tracker",
    page_icon="🤖",
    layout="wide"
)


# ================= CUSTOM CSS =================

st.markdown(
"""
<style>


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



[data-testid="stSidebar"] {


background:

linear-gradient(
180deg,
#141E30,
#243B55
);


}



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



.subtitle {


text-align:center;

font-size:22px;

color:#d9faff;


}



.card {


background:

rgba(255,255,255,0.12);


padding:25px;

border-radius:25px;


box-shadow:

0px 10px 30px rgba(0,0,0,0.4);


backdrop-filter:

blur(15px);


}



.metric {


font-size:45px;

font-weight:900;

text-align:center;


color:#00F5A0;


}



.stButton button {


background:

linear-gradient(
90deg,
#00F5A0,
#00D9F5
);


border-radius:20px;

font-weight:bold;

color:black;


}


</style>

""",
unsafe_allow_html=True
)



# ================= LOAD MODEL =================


@st.cache_resource
def load_model():

    model_path = "yolov8n.pt"


    if not os.path.exists(model_path):

        st.error(
            "YOLO model file missing!"
        )

        st.stop()


    return YOLO(model_path)



model = load_model()



# ================= HEADER =================


st.markdown(

"""
<div class="title">

🤖 AI Attendance Tracker

</div>


<div class="subtitle">

YOLOv8 • Computer Vision • Smart Attendance Detection

</div>

""",

unsafe_allow_html=True

)


st.write("")



# ================= SIDEBAR =================


with st.sidebar:


    st.markdown(

    """

    # ⚙️ AI Controls


    Adjust detection sensitivity.


    Higher confidence = fewer but more accurate detections.


    """

    )


    confidence = st.slider(

        "🎯 Detection Confidence",

        min_value=0.10,

        max_value=0.90,

        value=0.45,

        step=0.05

    )


    st.write(
        f"Current threshold: {confidence}"
    )



# ================= UPLOAD =================


uploaded_file = st.file_uploader(

    "📸 Upload Meeting Screenshot",

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




    # ================= DETECTION =================


    with st.spinner(
        "🧠 AI analyzing image..."
    ):


        results = model(

            img,

            conf=confidence

        )



        attendance = 0



        for result in results:


            for box in result.boxes:


                class_id = int(
                    box.cls[0]
                )


                # Person class

                if class_id == 0:


                    attendance += 1



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




    # ================= METRICS =================


    st.write("")


    a,b,c = st.columns(3)



    with a:


        st.markdown(

        f"""

        <div class="card">

        <div class="metric">

        {attendance}

        </div>


        👥 Attendance


        </div>

        """,

        unsafe_allow_html=True

        )



    with b:


        st.markdown(

        """

        <div class="card">

        <div class="metric">

        YOLOv8


        </div>


        🧠 Model


        </div>


        """,

        unsafe_allow_html=True

        )



    with c:


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


    Upload a meeting screenshot and let AI count attendees automatically.


    </div>


    """,

    unsafe_allow_html=True

    )
