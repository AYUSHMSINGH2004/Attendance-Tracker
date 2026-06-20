# 📌 AI Attendance Tracker using YOLOv8 & OpenCV

## 📖 Overview

AI Attendance Tracker is a computer vision-based attendance detection system that automatically detects and counts people from meeting screenshots or images using **YOLOv8 object detection** and **OpenCV**.

The project uses deep learning techniques to identify attendees, draw detection bounding boxes, and generate an automated attendance count without manual effort.

---

# 🚀 Features

- ✅ AI-based person detection
- ✅ Automated attendance counting
- ✅ YOLOv8 deep learning model integration
- ✅ OpenCV image processing
- ✅ Bounding box visualization
- ✅ Fast and lightweight detection
- ✅ Works with meeting screenshots/images
- ✅ Easy local deployment

---

# 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core development |
| OpenCV | Image processing |
| YOLOv8 | Person detection |
| Ultralytics | YOLO framework |
| Deep Learning | Object recognition |

---

# 📂 Project Structure

```
Attendance Tracker/
│
├── attendance_tracker.py     # Main application
├── meeting_grid.jpg          # Input meeting screenshot/image
├── yolov8n.pt                # YOLOv8 trained model
└── README.md                 # Project documentation
```

---

# ⚙️ Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/attendance-tracker.git

cd attendance-tracker
```

---

## Step 2: Install Dependencies

Make sure Python 3.8 or above is installed.

Install required libraries:

```bash
pip install opencv-python ultralytics
```

---

# ▶️ Running the Project

Run the application:

```bash
python attendance_tracker.py
```

The system will:

1. Load the input image
2. Detect people using YOLOv8
3. Draw bounding boxes
4. Count detected attendees
5. Display attendance result

---

# 🧠 Working Pipeline

```
Input Image
      |
      ↓
OpenCV Processing
      |
      ↓
YOLOv8 Object Detection
      |
      ↓
Person Detection
      |
      ↓
Attendance Calculation
      |
      ↓
Output with Detection Boxes
```

---

# 📸 Output

The model detects all visible people in the image and generates:

- Person detection boxes
- Number of detected attendees
- Processed output visualization


Example:

```
Total Attendance: 15
```

---

# 🤖 Model Details

This project uses:

## YOLOv8 Nano

YOLOv8n is selected because it provides:

- High-speed detection
- Low resource usage
- Real-time inference capability
- Good accuracy for person detection

---

# 🔮 Future Improvements

Planned upgrades:

- [ ] Face recognition attendance
- [ ] Student database integration
- [ ] Attendance history storage
- [ ] Web dashboard
- [ ] Live video meeting attendance
- [ ] Cloud deployment
- [ ] Multiple classroom/session support

---

# 🤝 Contribution Guidelines

Contributions are welcome.

Steps:

1. Fork this repository

2. Create a new branch:

```bash
git checkout -b feature-name
```

3. Commit changes:

```bash
git commit -m "Added new feature"
```

4. Push branch:

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Your Name**

GitHub:
```
https://github.com/your-username
```

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

---

## GitHub Repository Description

AI-powered attendance tracking system using YOLOv8 and OpenCV to automatically detect and count attendees from meeting images with real-time computer vision.

---

## GitHub Topics

```
python
opencv
yolo
yolov8
computer-vision
deep-learning
artificial-intelligence
object-detection
machine-learning
attendance-system
ai-project
```
