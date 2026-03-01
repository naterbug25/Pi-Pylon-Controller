1. System Overview
The Pi-Pylon-Controller is an industrial-grade edge computer vision system designed to run on a Raspberry Pi (ARM64). It features real-time image acquisition, local deep learning model training (Transfer Learning), dual-Region of Interest (ROI) processing, and industrial PLC handshaking via EtherNet/IP.

To ensure the user interface remains highly responsive during heavy AI inference or network timeouts, the system strictly relies on a Dual-Process Architecture.

2. Architecture & Inter-Process Communication (IPC)
2.1 The Multi-Process Split
The system is divided into two primary isolated processes:

Main Process (HMI & Orchestrator): Manages the graphical user interface, user inputs, and system bootstrapping.

Daemon Process (Vision Engine): A high-priority background loop that handles camera hardware, neural network inference, and PLC communications.

2.2 Shared State Management
Communication between the two processes occurs via a multiprocessing.Manager().dict(). This shared dictionary acts as the single source of truth for the system.

Flags: trigger_request, reload_request, cam_reload.

Telemetry: result_status, training_progress, history (list), heartbeat (integer counter).

Hardware IO: io_in (Trigger), io_out (Pass, Fail, Running).

Configuration: active_program, cam_source, plc_ip, basler_ip, class_configs (dict of class names and thresholds).

Spatial Data: search_roi and crop_roi (bounding box coordinates stored as percentages from 0.0 to 1.0).

2.3 Atomic File Swapping
To prevent the HMI from reading a corrupted or partially-written video frame, the Vision Engine writes live camera feeds to a temporary file (live_buffer.tmp.jpg) and uses a native OS atomic replace operation (os.replace) to overwrite the active file (live_buffer.jpg).

3. Core Modules & Functionality
3.1 The Orchestrator (Bootstrapper)
Directory Management: Upon startup, it ensures the existence of required directories (models/, config/, and dataset/Class_0 through Class_4).

State Hydration: Reads config/settings.json to load the last known IP addresses, ROI configurations, and active program, injecting them into the shared state.

Graceful Shutdown: Implements a try/finally block. When the HMI is closed, it explicitly terminates the background Vision Engine process using p.terminate() and p.join() to prevent "zombie" processes from keeping the camera or web ports locked.

Wayland Override: Forces the Qt framework to use the xcb platform plugin to maintain compatibility with newer Raspberry Pi OS versions.

3.2 The Vision Engine (Hardware & AI Loop)
Runs a continuous while True loop that performs the following:

Heartbeat & Sync: Increments a heartbeat counter every iteration. Every 40-50 ticks, it spins off an asynchronous, non-blocking thread to read/write tags to the PLC using the pycomm3 library.

Camera Acquisition: Supports USB Webcams (via OpenCV) and Basler GigE cameras (via pypylon). Uses a "Latest Image Only" grabbing strategy to prevent buffer latency.

Dual-ROI Implementation: * Draws two distinct bounding boxes on the live feed: a Yellow Box for the Search Area, and a Red Box for the Training Area.

Atomic Trigger Handshake: When trigger_request is True, it immediately sets it back to False (preventing race conditions).

If in RUN mode: It crops the image to the Search ROI, runs AI inference, and outputs PASS/FAIL to the PLC.

If in TRAIN mode: It crops the image to the Training ROI, freezes the frame by saving it as temp_capture.jpg, and waits for the HMI to categorize it.

MJPEG Streaming: Hosts a lightweight Flask server on port 5000 that serves the atomic live_buffer.jpg to any web browser on the network.

3.3 The HMI Application (PyQt6 UI)
A dark-themed industrial dashboard updating at 30 FPS via a QTimer.

Live Video Panel: Displays the live_buffer.jpg or freezes on temp_capture.jpg after a trigger in TRAIN mode.

Status Header & Heartbeat: A large text label showing the current mode/result, flanked by a custom-drawn Pilot Light that flashes based on the Vision Engine's heartbeat integer.

IO Pilot Lights: Custom UI widgets mapping directly to the PLC's input (Trigger) and outputs (Pass, Fail, Running).

Tabbed ROI Controllers: Two separate tabs containing horizontal sliders to adjust x_min, x_max, y_min, and y_max for the Search and Crop ROIs.

Class Configurator: Text boxes to rename the 5 standard classes and spinboxes to adjust the decimal confidence threshold (0.0 to 1.0) required to trigger a PASS.

Job Management: A List Widget containing saved programs. Allows creating new jobs and deleting old ones. Changes here trigger a model reload in the Vision Engine.

Forensic Logger: A scrolling list showing the historical results. Displays the winning class in bright text, followed by an indented, dimmed string showing the exact percentage breakdown of all 5 classes for deep-dive diagnostics.

Settings Dialog: A popup to configure the Camera Source (Webcam/Basler), Basler IP, PLC IP, and IO Mode (PLC/GPIO).

3.4 The Local Trainer (Machine Learning)
Executes within a background thread spawned by the HMI to prevent UI freezing during model compilation.

Architecture: Uses a MobileNetV2 base architecture imported from TensorFlow/Keras. The ImageNet weights are frozen. A GlobalAveragePooling2D and a Dense softmax output layer are appended.

Dynamic Head: Automatically determines the number of output classes based on which dataset/Class_X folders actually contain valid images. Requires at least 2 populated classes.

Pipeline: Loads images at 224x224 resolution with a 20% validation split.

HMI Callback: Implements a custom tf.keras.callbacks.Callback to update the shared training_progress integer at the end of each epoch, which drives the HMI progress bar.

Native Keras Format: Saves the final compiled model as a .keras file to bypass flatbuffer conversion issues common to ARM64 TFLite deployments.

4. File Schema & Data Structures
4.1 Folder Structure Layout
Plaintext
/home/pi/Pi-Pylon-Controller/
├── main.py
├── vision_engine.py
├── hmi_app.py
├── trainer.py
├── config/
│   └── settings.json
├── models/
│   ├── Part_A.keras
│   └── [Custom_Job_Names].keras
└── dataset/
    ├── Class_0/  (e.g., Target Object)
    ├── Class_1/  (e.g., Background/Negative)
    ├── Class_2/
    ├── Class_3/
    └── Class_4/
4.2 Settings JSON Schema
The configuration file dictates the state upon boot.

JSON
{
    "active_program": "Part_A",
    "program_list": ["Part_A", "Job_2"],
    "io_mode": "PLC",
    "cam_source": "Basler",
    "basler_ip": "192.168.1.50",
    "plc_ip": "192.168.1.10",
    "crop_roi": {"x_min": 0.25, "x_max": 0.75, "y_min": 0.25, "y_max": 0.75},
    "search_roi": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
    "class_configs": {
        "0": {"name": "Pass_Part", "threshold": 0.85},
        "1": {"name": "Fail_Background", "threshold": 0.85}
    }
}
5. Critical Execution Logic Requirements
If rebuilding this system, the following engineering rules must be followed to prevent catastrophic failures:

Trigger Atomic Reset: The Vision Engine must reset trigger_request to False the exact moment it detects it. Delaying this reset will cause the HMI to stack trigger requests, locking the software.

Threaded IO: PLC communication (pycomm3) will hang the main Vision Engine loop if the physical PLC is disconnected or slow. PLC syncs must be placed inside a non-blocking threading.Thread(target=..., daemon=True).

UI Element Layouts: PyQt6 Layouts must be explicitly assigned to their parent widgets (e.g., tab.setLayout(layout)) or they will fail to render on screen without throwing an error.