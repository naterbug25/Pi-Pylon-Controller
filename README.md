Pi-Pylon-Controller: System Documentation
The Pi-Pylon-Controller is an industrial-grade edge vision system designed for Raspberry Pi (ARM64). it provides a complete pipeline for image acquisition, local deep learning training, and real-time inference with PLC handshaking via EtherNet/IP.

1. System Architecture
The system utilizes a Multi-Process Architecture to decouple time-critical vision tasks from the User Interface.

Core Components
Orchestrator (main.py): The system entry point. It initializes a multiprocessing.Manager to create a shared memory state accessible by both the UI and the Vision Engine. It also handles persistent configuration loading from config/settings.json.

Vision Engine (vision_engine.py): A high-priority background process responsible for:

Hardware abstraction for Basler GigE and USB cameras.

Real-time Neural Network inference using Native Keras.

Industrial communication with Allen-Bradley/Logix PLCs.

Hosting a Flask Web Server for remote MJPEG video streaming.

HMI Application (hmi_app.py): A PyQt6 interface providing real-time telemetry, job management, and dataset collection tools.

Local Trainer (trainer.py): A training module using Transfer Learning (MobileNetV2) to build custom classification models directly on the edge hardware.

2. Functional Specifications
2.1 Camera Acquisition
Basler Integration: Uses pypylon. Supports IP-based addressing. Employs a LatestImageOnly strategy to ensure inference is always performed on the most recent frame, preventing buffer backlog.

USB Fallback: Uses OpenCV (V4L2). Defaults to index 0.

2.2 Deep Learning Pipeline
The system is built on TensorFlow 2.16+ and NumPy 2.x. To bypass common TFLite conversion bugs on ARM64, it uses Native Keras (.keras) format for both training and inference.

Inference Engine: 1.  Captures 640x480 frame.
2.  Resizes to 224x224 and converts to RGB.
3.  Applies mobilenet_v2.preprocess_input.
4.  Runs model.predict() and extracts the Softmax probability array.

Training Engine:

Uses MobileNetV2 with frozen ImageNet weights.

Appends a GlobalAveragePooling2D and a Dense output layer.

Dynamically adjusts the output head based on the number of populated Class_X folders.

Implements a Progress Callback to update the HMI loading bar in real-time.

2.3 Industrial I/O & Handshaking
The system uses pycomm3 for EtherNet/IP communication with a Logix Driver.

Trigger Input: Monitors PLC Tag Vision_Trigger. A True state triggers a single inspection.

Result Output: Writes True/False to PLC Tags Vision_Pass and Vision_Fail.

Handshake Logic: The RUNNING bit indicates the engine is currently processing.

3. HMI Features (PyQt6)
3.1 Monitoring & Control
Status Header: A high-visibility 32px label displaying the current result (e.g., "Part_A: PASS 98%").

Live View: Renders 30 FPS video with overlay support.

Pilot Lights: Custom-drawn UI elements reflecting the state of TRIGGER, PASS, and FAIL bits.

Inspection Log: A forensic logger showing the winning class and the confidence scores of all secondary classes for every trigger.

3.2 Machine Setup
Program Manager: Create, select, and delete "Jobs." Each job maintains its own .keras model.

Class Configurator: * Custom naming for up to 5 classes.

Threshold Adjustment: Per-class decimal threshold (0.0 to 1.0) to determine PASS/FAIL criteria.

Training Mode: Shifts the HMI into "Collection" state. Triggers now "freeze" the frame, allowing the user to assign the image to a specific Class folder.

4. Operational Requirements
4.1 Folder Structure
Plaintext

Pi-Pylon-Controller/
├── main.py             # Entry Point
├── vision_engine.py    # Hardware/AI Loop
├── hmi_app.py          # UI
├── trainer.py          # ML Training
├── config/
│   └── settings.json   # Persistent JSON
├── models/
│   └── *.keras         # Compiled Models
└── dataset/
    ├── Class_0/        # Image Samples
    └── Class_1/        # Image Samples
4.2 Environment
OS: Raspberry Pi OS 64-bit (Bookworm).

Python: 3.11+.

Key Libraries: tensorflow>=2.16, numpy>=2.0, opencv-python-headless, PyQt6, pycomm3, pypylon.

5. Deployment Guide
Initialize the environment using a virtual environment (.venv).

Install system dependencies for Qt6 and OpenCV (libqt6gui6, libatlas-base-dev).

Configure PLC IP and Basler IP in the HMI Settings.

Capture at least 15 images per class in TRAIN mode.

Execute TRAIN LOCAL and verify the progress bar reaches 100%.

Switch to RUN mode to begin PLC-triggered inspections.