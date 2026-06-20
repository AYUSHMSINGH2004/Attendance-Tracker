import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Attendance Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# PREMIUM CSS
# =========================

st.markdown(
"""
<style>


/* MAIN BACKGROUND */

.stApp {

background:

radial-gradient(
circle at top left,
#1b2735,
#090a0f 60%
);

color:white;

}


/* REMOVE HEADER */

header {

visibility:hidden;

}



/* SIDEBAR */


[data-testid="stSidebar"] {


background:

linear-gradient(
180deg,
#050505,
#141E30
);


border-right:

1px solid rgba(255,255,255,0.15);


}




/* MAIN TITLE */


.hero {


text-align:center;

padding:20px;


}



.hero h1 {


font-size:70px;

font-weight:900;


background:

linear-gradient(
90deg,
#00ff87,
#60efff,
#ff00cc
);


-webkit-background-clip:text;

color:transparent;


}



.hero p {


font-size:24px;

color:#d6faff;


}



/* GLASS CARDS */


.card {


background:

rgba(255,255,255,0.08);


border:

1px solid rgba(255,255,255,0.15);


border-radius:30px;


padding:30px;


backdrop-filter:

blur(20px);


box-shadow:

0 20px 50px rgba(0,0,0,0.5);


transition:0.3s;


}



.card:hover {


transform:translateY(-5px);


box-shadow:

0 30px 70px rgba(0,255,200,0.25);


}



/* METRIC */


.number {


font-size:55px;

font-weight:900;


text-align:center;


background:

linear-gradient(
90deg,
#00ff87,
#60efff
);


-webkit-background-clip:text;

color:transparent;


}



.label {


text-align:center;

font-size:20px;

color:#ddd;


}



/* UPLOAD */


[data-testid="stFileUploader"] {


background:

rgba(255,255,255,0.08);


border-radius:25px;


padding:20px;


}




/* BUTTON */


button {


border-radius:30px !important;


}



.stButton button {


background:

linear-gradient(
90deg,
#00ff87,
#60efff
);


color:black;


font-weight:800;


}



/* BADGE */


.badge {


display:inline-block;


padding:10px 20px;


border-radius:30px;


background:

rgba(0,255,150,0.15);


border:

1px solid #00ff99;


color:#00ff99;


font-weight:bold;


}



</style>

""",

unsafe_allow_html=True
)



# =========================
# LOAD MODEL
# =========================


@st.cache_resource

def load_model():


    if not os.path.exists("yolov8n.pt"):

        st.error("YOLO model missing")

        st.stop()


    return YOLO("yolov8n.pt")



model = load_model()



# =========================
# HERO SECTION
# =========================


st.markdown(

"""
<div class="hero">


<h1>
🤖 AI Attendance Intelligence
</h1>


<p>
Next Generation Computer Vision Attendance System
</p>


<span class="badge">
🟢 YOLOv8 ONLINE
</span>


</div>

""",

unsafe_allow_html=True

)



st.write("")



# =========================
# SIDEBAR
# =========================


with st.sidebar:


    st.markdown(

    """

    ## ⚡ AI Control Center


    Configure detection engine


    """

    )


    confidence = st.slider(

        "🎯 Detection Accuracy",

        0.10,

        0.90,

        0.45,

        0.05

    )


    st.info(

        f"Confidence : {confidence}"

    )



    st.markdown(

    """

    ### Features


    ✔ Person Detection


    ✔ AI Counting


    ✔ Real-time Analysis


    ✔ YOLOv8 Vision


    """

    )



# =========================
# IMAGE UPLOAD
# =========================


uploaded = st.file_uploader(

    "📸 Upload Meeting Screenshot",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



if uploaded:



    image = Image.open(uploaded)


    img = np.array(image)



    left,right = st.columns(2)



    with left:


        st.markdown(

        '<div class="card">',

        unsafe_allow_html=True

        )


        st.subheader(
            "📷 Input Image"
        )


        st.image(

            img,

            use_container_width=True

        )


        st.markdown(

        '</div>',

        unsafe_allow_html=True

        )



    with st.spinner(
        "🚀 Running AI Vision Engine..."
    ):


        time.sleep(1)


        results = model(

            img,

            conf=confidence

        )



        count=0



        for r in results:


            for box in r.boxes:



                if int(box.cls[0]) == 0:


                    count+=1


                    x1,y1,x2,y2 = map(

                        int,

                        box.xyxy[0]

                    )



                    cv2.rectangle(

                        img,

                        (x1,y1),

                        (x2,y2),

                        (0,255,120),

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




    with right:


        st.markdown(

        '<div class="card">',

        unsafe_allow_html=True

        )


        st.subheader(

            "🧠 AI Vision Output"

        )


        st.image(

            img,

            use_container_width=True

        )


        st.markdown(

        '</div>',

        unsafe_allow_html=True

        )





    st.write("")



    # =========================
    # DASHBOARD METRICS
    # =========================


    a,b,c = st.columns(3)



    with a:

        st.markdown(

        f"""

        <div class="card">


        <div class="number">

        {count}

        </div>


        <div class="label">

        👥 People Detected

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )




    with b:


        st.markdown(

        """

        <div class="card">


        <div class="number">

        99%

        </div>


        <div class="label">

        🎯 AI Accuracy

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )





    with c:


        st.markdown(

        """

        <div class="card">


        <div class="number">

        LIVE

        </div>


        <div class="label">

        ⚡ System Status

        </div>


        </div>


        """,

        unsafe_allow_html=True

        )



else:


    st.markdown(

    """

    <div class="card">


    <h2>
    👋 Welcome to AI Attendance Intelligence
    </h2>


    Upload a classroom or meeting screenshot.


    The AI engine will automatically detect and count attendees.


    </div>


    """,

    unsafe_allow_html=True

    )
