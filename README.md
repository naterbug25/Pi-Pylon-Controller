# Pi-Pylon Controller (YOLO Edition)

An open-source, industrial edge AI vision system built for Raspberry Pi and Windows. This system utilizes the **YOLOv8** object detection architecture to locate and verify objects across an entire camera field of view, completely eliminating the need for strict, static bounding boxes. 

It features a built-in PyQt6 Human-Machine Interface (HMI) with live drag-and-drop annotation, on-device model training, Allen-Bradley PLC integration, and support for industrial Basler GigE cameras.



## Features
* **YOLOv8 Object Detection**: Natively searches the full frame (scale and location invariant).
* **Cross-Platform**: Runs on Raspberry Pi (deployment) and Windows PCs with NVIDIA GPUs (rapid training).
* **Live HMI Annotation**: Draw bounding boxes directly on the live camera feed to teach the AI new parts.
* **On-Device Training**: Train custom neural networks locally without needing the cloud.
* **PLC Integration**: Communicates directly with Allen-Bradley PLCs via Ethernet/IP (Pycomm3).
* **Industrial Camera Support**: Built-in support for Basler GigE Machine Vision cameras (via Pypylon) or standard USB Webcams.
* **MJPEG Streaming**: Serves a live camera feed over the network via a lightweight Flask server.

---

## Hardware Requirements
To fully recreate this system, you will need:
1. **Compute**: A Raspberry Pi 4/5 (for edge deployment) OR a Windows PC with an NVIDIA GPU (for rapid training).
2. **Camera**: A standard USB Webcam OR a Basler GigE Industrial Camera.
3. **Trigger (Optional)**: An Allen-Bradley PLC (e.g., Micro800, CompactLogix) for industrial triggering, or simply use the HMI "Trigger" button.

---

## Installation & Setup

Because this system requires complex math libraries, it is highly recommended to install the dependencies inside a Python Virtual Environment.

### 1. Clone the Repository
Clone or download these four core files into a single project directory:
* `main.py`
* `vision_engine.py`
* `hmi_app.py`
* `trainer.py`

### 2. Create a Virtual Environment
Open your terminal or command prompt in the project directory:
```bash
python -m venv venv
Activate the environment:

Windows: .\venv\Scripts\activate

Linux/Raspberry Pi: source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file in your directory with the following contents. (Note: The numpy version restriction is critical to prevent a known compatibility crash with OpenCV/Matplotlib).

If using a Windows PC with an NVIDIA GPU (Recommended for Training):

Plaintext
--extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
torch
torchvision
torchaudio
PyQt6
opencv-contrib-python-headless>=4.10
pycomm3
flask
pypylon
ultralytics
numpy<2
If using a Raspberry Pi (CPU Only):

Plaintext
PyQt6
opencv-contrib-python-headless>=4.10
pycomm3
flask
pypylon
ultralytics
numpy<2
Run the installation:

Bash
pip install -r requirements.txt
Note on Basler Cameras: If you are using a Basler camera, you must also download and install the official "pylon Camera Software Suite" for your respective OS from the Basler website.

System Architecture
The software is divided into four modular components that run simultaneously using Python's multiprocessing to ensure the UI never freezes while the camera is processing frames.

main.py (The Orchestrator): Bootstraps the application, creates the folder structures (models/, dataset/, etc.), and initializes the shared memory dictionary that allows the UI and the Vision Engine to talk to each other.

vision_engine.py (The AI Core): Runs in the background. It manages the camera hardware, listens for PLC triggers, runs the YOLO mathematical inference, draws the bounding boxes, and serves the Flask video stream.

hmi_app.py (The Interface): The PyQt6 graphical interface. Handles job management, confidence threshold adjustments, and the custom mouse-drawing tools for annotating images.

trainer.py (The Compiler): Generates YOLO data.yaml files dynamically and orchestrates the local training pipeline, moving the finished .pt weights file to the correct directory upon completion.

How to Use the System
Phase 1: Teaching the AI (Annotation)
Run the application: python main.py

Click MODE on the left panel to switch to TRAIN mode.

Place your target object in front of the camera.

Click TRIGGER. The video feed will freeze.

Click and drag your mouse to draw a tight green bounding box around the object.

Select the appropriate Class (e.g., Class 0) from the dropdown and click Save Annotation.

Move the object, rotate it, or alter the lighting, and repeat steps 4-6 until you have ~20-50 annotated images.

Phase 2: Training the Model
Once you have captured your dataset, click TRAIN YOLO.

The progress bar will fill.

On a Windows PC with a GPU, this takes ~2 minutes.

On a Raspberry Pi CPU, this takes ~1-2 hours.

The system trains for 50 epochs at a 320x320 resolution for optimized speed. When complete, the status will say SUCCESS.

Phase 3: Running Inspections
Click MODE to return to RUN mode.

Place the object anywhere in the camera's field of view.

Click TRIGGER (or fire the Vision_Trigger tag from your PLC).

The Vision Engine will scan the image, draw a bounding box around the detected object, and output a PASS or FAIL based on the Confidence Thresholds set in the right-hand panel.

Pro-Tip: The "Train Heavy, Run Light" Workflow
For the best performance, do not train models on the Raspberry Pi.

Run the system on your Pi to collect and annotate your images.

Copy the dataset folder to a Windows PC equipped with an NVIDIA GPU.

Run main.py on the PC and click TRAIN YOLO (finishes in minutes).

Copy the resulting models/Part_A.pt file back to your Raspberry Pi.

The Pi can now run live inspections at high speeds using the heavy math computed by the PC.