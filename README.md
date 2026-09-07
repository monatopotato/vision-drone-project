# 🛸 Autonomous Vision-Guided Tello Drone Assistant

An AI-powered autonomous flight and object detection system for Ryze / DJI Tello EDU drones. Powered by **PyTorch MobileNetV3 SSDLite**, real-time visual servoing, and a natural language terminal chat assistant.

---

## 🌟 Key Features

- **🤖 Natural Language Chat Assistant**: Type natural instructions in terminal (e.g. *"go to a ball and hover above it"*, *"once you go to a computer, land"*).
- **🎯 Real-Time Bounding-Box Object Detection**: Powered by **PyTorch MobileNetV3 SSDLite** pre-trained on COCO everyday objects (**sports ball**, **laptop/computer**, **bottle**, **cup**, **cell phone**, **chair**, **book**, etc.) running at **30–60+ FPS on CPU**.
- **📐 Visual Servoing & Centering Loop**: Aligns drone heading (yaw) and altitude while monitoring bounding-box area ratios ($A_{box} / A_{frame}$) to navigate autonomously toward target objects.
- **🏠 High-Accuracy Room Classification**: MobileNetV3 transfer learning model fine-tuned for indoor room recognition with **88.78%+ validation accuracy** and **12.27 ms/frame** latency.
- **💻 Dual Execution Modes**: Works connected to a live Tello drone Wi-Fi feed or in **PC Webcam Simulation Mode** for testing without hardware.

---

## 📐 System Architecture

```text
  +--------------------------------+
  | Terminal Chat Input            | <--- User Types: "go to a ball and hover"
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  | NLP Intent Parser              | ---> Extracts: Target='sports ball', Action='HOVER'
  | (command_parser.py)            |
  +--------------------------------+
                  |
                  v
  +--------------------------------+      +-------------------------------+
  | Autonomous Visual Servoing     | <==> | PyTorch MobileNetV3 SSDLite   |
  | Engine (autonomous_tracker.py) |      | Real-Time Bounding Boxes      |
  +--------------------------------+      +-------------------------------+
                  |
    [Yaw / Pitch / Height / Speed Commands]
                  v
  +--------------------------------+
  | Tello EDU Drone Flight Control |
  +--------------------------------+
```

---

## 📁 Repository Structure

```text
vision-drone-project/
├── drone_assistant_chat.py       # Main Autonomous Natural Language Vision Assistant
├── command_parser.py             # NLP Intent & Target Synonyms Extractor
├── autonomous_tracker.py         # Visual Servoing Alignment & Distance Controller
├── real_time_object_detector.py  # PyTorch MobileNetV3 SSDLite Object Detector (Webcam + API)
├── drone_object_detector_ssd.py  # Tello Drone Bounding-Box Object Detector with HUD
├── test_webcam_object_detector.py# Webcam Test Interface for Object Recognition
├── room_classifier.py            # Indoor Room Classification PyTorch Module
├── train_room_model.py           # 2-Stage MobileNetV3 Transfer Learning Trainer
├── train_object_model.py         # Everyday Object Classifier Trainer
├── download_object_dataset.py    # Automated Everyday Object Dataset Loader
├── interactive_drone_cli.py      # Interactive Text CLI Flight Controller
├── up_down_flight.py             # Simple Timed Maneuver Script
├── tello.py                      # Tello UDP Socket Communication Wrapper
├── stats.py                      # Tello Command Telemetry Logger
├── requirements.txt              # Project Dependencies
└── README.md                     # Project Documentation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.9+ installed on Windows, macOS, or Linux.
- Open Wi-Fi connection for Tello drone (`TELLO-XXXXXX`).

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/monatopotato/vision-drone-project.git
cd vision-drone-project
pip install -r requirements.txt
```

---

## 🕹️ Usage Guide

### 1. Autonomous AI Drone Assistant (Terminal Chat)
Connect to Tello Wi-Fi (or run on PC for webcam simulation) and launch:
```bash
python drone_assistant_chat.py
```
**Example Commands to Type:**
- `"go to a ball and hover above it"`
- `"once you go to a computer, land"`
- `"find a bottle and hover"`
- `"go to phone and land"`
- `"takeoff"`
- `"land"`

---

### 2. Real-Time Everyday Object Detector (Bounding Boxes)
Run bounding-box object detection on your **webcam** (no drone needed):
```bash
python real_time_object_detector.py
```
Run bounding-box object detection on **Tello Drone Live Feed**:
```bash
python drone_object_detector_ssd.py
```
- **`s`**: Save snapshot of detected objects with bounding box metadata.
- **`f`**: Execute autonomous hover survey flight.
- **`q`**: Exit application.

---

### 3. Interactive Text Flight CLI
Control drone movement interactively with simple text commands:
```bash
python interactive_drone_cli.py
```
- **`takeoff`** / **`land`**: Launch or land safely.
- **`up`** / **`down`**: Move up/down by 30 cm (e.g. `up 50`, `down 20`).
- **`forward`** / **`back`** / **`left`** / **`right`**: Tactical movement.
- **`turn left`** / **`turn right`**: Rotate heading by 45 degrees.
- **`battery`**: Query current drone battery percentage.

---

## 📊 Performance Benchmarks

| Feature | Model Architecture | Frame Rate / Latency | Accuracy / Performance |
| :--- | :--- | :--- | :--- |
| **Object Bounding-Box Detection** | PyTorch MobileNetV3 SSDLite | **30–60+ FPS (CPU)** | Pre-trained on 80 COCO categories |
| **Indoor Room Classification** | Fine-tuned MobileNetV3-Large | **12.27 ms/frame (81.5 FPS)** | **88.78% Validation Accuracy** |

---

## 📜 License
Licensed under the [MIT License](LICENSE).
