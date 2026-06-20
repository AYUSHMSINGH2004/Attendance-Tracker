# 🤖 AI Attendance Tracker using YOLOv8 & OpenCV

![AI Attendance Tracker](https://img.shields.io/badge/AI-YOLOv8-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)

An AI-powered automated attendance tracking system that uses **YOLOv8 object detection** and **OpenCV** to detect and count people from classroom/meeting screenshots.

The project provides a modern interactive dashboard where users can upload an image, run AI-based person detection, visualize detected attendees, and instantly calculate attendance.

---

# 🌐 Live Demo

🚀 Try the deployed application:

https://attendance-tracker-qtjbpju7v5jjsapptpnrxtw.streamlit.app/

---

# ✨ Features

## 🧠 AI Vision

- YOLOv8 based person detection
- Deep learning powered attendance estimation
- Real-time image analysis


## 📊 Smart Dashboard

- Modern AI themed interface
- Interactive confidence control
- Detection visualization
- Attendance statistics


## ⚡ Performance

- Fast inference
- Lightweight YOLOv8 Nano model
- Cloud deployed application


---

# 🖥️ Application Preview

The system workflow:

```
Upload Meeting Image

        ↓

YOLOv8 Detection Engine

        ↓

Person Identification

        ↓

Bounding Box Generation

        ↓

Attendance Count
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| YOLOv8 | Object detection |
| OpenCV | Image processing |
| Ultralytics | YOLO framework |
| Streamlit | Web application |
| NumPy | Data processing |
| Pillow | Image handling |

---

# 📂 Project Structure

```
Attendance-Tracker/

│
├── app.py                    # Streamlit application
├── attendance_tracker.py     # Local detection script
├── yolov8n.pt                # YOLOv8 model
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── meeting_grid.jpg          # Sample input
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AYUSHMSINGH2004/Attendance-Tracker.git

cd Attendance-Tracker
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

Deployment flow:

```
GitHub Repository

        ↓

Streamlit Cloud

        ↓

Live AI Web Application
```

Live URL:

```
https://attendance-tracker-qtjbpju7v5jjsapptpnrxtw.streamlit.app/
```

---

# 🧠 How It Works

1. User uploads a meeting/classroom image

2. Image is processed using OpenCV

3. YOLOv8 detects human objects

4. Detected people are highlighted

5. System calculates total attendance

---

# 🎯 Model Information

This project uses:

## YOLOv8 Nano

Advantages:

- Fast detection speed
- Low computational requirements
- Suitable for real-time applications
- Optimized for deployment

---

# 📈 Future Enhancements

Possible improvements:

- [ ] Face recognition based attendance
- [ ] Student database integration
- [ ] Attendance history dashboard
- [ ] Export attendance reports
- [ ] Live camera attendance
- [ ] User authentication
- [ ] Cloud database integration

---

# 🤝 Contributing

Contributions are welcome.

Steps:

```bash
git clone repository

git checkout -b feature-name

git commit -m "Added feature"

git push origin feature-name
```

Create a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Ayush M Singh**

GitHub:

```
https://github.com/AYUSHMSINGH2004
```

---

# ⭐ Support

If you like this project, consider giving it a star ⭐
