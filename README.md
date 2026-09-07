# 🛸 Autonomous Vision-Guided Tello Drone Assistant

An AI-powered autonomous flight and object detection system for Ryze / DJI Tello EDU drones. Powered by **PyTorch MobileNetV3 SSDLite**, real-time visual servoing, and a natural language terminal chat assistant.

---

## 🌟 Key Features

- **🤖 Natural Language Chat Assistant**: Type natural instructions in terminal (e.g. *"go to a ball and hover above it"*, *"once you go to a computer, land"*).
- **🎯 Real-Time Bounding-Box Object Detection**: Powered by **PyTorch MobileNetV3 SSDLite** pre-trained on COCO everyday objects (**sports ball**, **laptop/computer**, **bottle**, **cup**, **cell phone**, **chair**, **book**, etc.) running at **30–60+ FPS on CPU**.
- **📐 Visual Servoing & Centering Loop**: Aligns drone heading (yaw) and altitude while monitoring bounding-box area ratios ($A_{box} / A_{frame}$) to navigate autonomously toward target objects.
- **🏠 High-Accuracy Room Classification**: MobileNetV3 transfer learning model fine-tuned for indoor room recognition with **88.78%+ validation accuracy** and **12.27 ms/frame** latency.
- **🎮 Main Launcher (`main.py`)**: One-command interactive CLI menu to run any mode instantly.

---

## 📁 Clean Package Architecture

```text
vision-drone-project/
├── main.py                       # 🚀 Interactive Project Launcher
├── src/                          # 📦 Modular Source Code
│   ├── assistant/                # 🤖 Autonomous Chat Assistant & Visual Servoing
│   │   ├── drone_assistant_chat.py
│   │   ├── command_parser.py
│   │   └── autonomous_tracker.py
│   ├── detection/                # 🎯 PyTorch MobileNetV3 Object Detection
│   │   ├── real_time_object_detector.py
│   │   ├── drone_object_detector_ssd.py
│   │   ├── test_webcam_object_detector.py
│   │   └── object_classifier.py
│   ├── classification/           # 🏠 Indoor Room Classifier & Model Trainers
│   │   ├── room_classifier.py
│   │   ├── train_room_model.py
│   │   ├── train_object_model.py
│   │   ├── download_dataset.py
│   │   ├── download_object_dataset.py
│   │   ├── eval_model.py
│   │   └── benchmark_backbones.py
│   ├── drone/                    # 🛸 Tello SDK UDP Communication & Telemetry
│   │   ├── tello.py
│   │   ├── stats.py
│   │   └── camera_stream.py
│   └── cli/                      # 🎮 Interactive Text CLI & Flight Scripts
│       ├── interactive_drone_cli.py
│       ├── up_down_flight.py
│       ├── simple_flight.py
│       └── flight_with_camera.py
├── app.py                        # 🌐 Flask Web Dashboard App
├── requirements.txt              # 📌 Project Dependencies
└── README.md                     # 📖 Project Documentation
```

---

## 🚀 Quick Start

```bash
# 1. Clone Repository
git clone https://github.com/monatopotato/vision-drone-project.git
cd vision-drone-project

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch Interactive Menu
python main.py
```

---

## 🕹️ Quick Commands

- **Run AI Assistant**: `python -m src.assistant.drone_assistant_chat`
- **Run Object Bounding-Box Detector**: `python -m src.detection.drone_object_detector_ssd`
- **Test Webcam Detector**: `python -m src.detection.real_time_object_detector`
- **Run Text Flight CLI**: `python -m src.cli.interactive_drone_cli`

---

## 📜 License
Licensed under the [MIT License](LICENSE).
