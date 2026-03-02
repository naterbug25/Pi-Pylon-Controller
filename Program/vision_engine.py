import cv2, time, os, threading
from pycomm3 import LogixDriver
from flask import Flask, Response

try:
    from ultralytics import YOLO
    YOLO_SUPPORT = True
except:
    YOLO_SUPPORT = False

try:
    from pypylon import pylon
    BASLER_SUPPORT = True
except:
    BASLER_SUPPORT = False

class VisionEngine:
    def __init__(self, state):
        self.state = state
        self.model = None
        self.camera = None

    def gen_frames(self):
        while True:
            if os.path.exists("live_buffer.jpg"):
                with open("live_buffer.jpg", "rb") as f:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n')
            time.sleep(0.05)

    def init_camera(self):
        if self.camera:
            try:
                if hasattr(self.camera, 'release'): self.camera.release()
                if hasattr(self.camera, 'Close'): self.camera.Close()
            except: pass
        if self.state['cam_source'] == 'Basler' and BASLER_SUPPORT:
            try:
                info = pylon.DeviceInfo(); info.SetIpAddress(self.state['basler_ip'])
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
                self.camera.Open(); self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except: self.camera = None
        else:
            self.camera = cv2.VideoCapture(0)

    def run_loop(self):
        app = Flask(__name__)
        @app.route('/')
        def video(): return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True).start()

        self.init_camera()
        self.load_resources()
        
        while True:
            self.state['heartbeat'] += 1
            if self.state.get('cam_reload'): self.init_camera(); self.state['cam_reload'] = False
            if self.state.get('reload_request'): self.load_resources(); self.state['reload_request'] = False

            if self.state['io_mode'] == 'PLC' and self.state['heartbeat'] % 40 == 0:
                threading.Thread(target=self.sync_plc, daemon=True).start()

            frame = None
            if self.state['cam_source'] == 'Basler' and self.camera:
                try:
                    res = self.camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
                    if res.GrabSucceeded(): frame = res.Array; res.Release()
                except: pass
            elif self.camera:
                ret, frame = self.camera.read()

            if frame is None: time.sleep(0.01); continue

            # Save clean frame for HMI live feed
            disp = frame.copy()
            cv2.imwrite("live_buffer.tmp.jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            os.replace("live_buffer.tmp.jpg", "live_buffer.jpg")

            # ATOMIC TRIGGER
            if self.state.get('trigger_request'):
                self.state['trigger_request'] = False
                self.state['io_out']['RUNNING'] = True
                
                if self.state['mode'] == 'RUN':
                    self.perform_inspection(frame)
                else:
                    cv2.imwrite("temp_capture.jpg", frame)
                    self.state['last_captured_frame'] = True
                    self.state['result_status'] = "DRAW BOX & SAVE ANNOTATION"
                
                self.state['io_out']['RUNNING'] = False
            time.sleep(0.001)

    def load_resources(self):
        p = f"models/{self.state['active_program']}.pt"
        if os.path.exists(p) and YOLO_SUPPORT:
            try: self.model = YOLO(p)
            except: self.model = None
        else:
            self.model = None

    def sync_plc(self):
        try:
            with LogixDriver(self.state['plc_ip']) as plc:
                if plc.read('Vision_Trigger').value: self.state['trigger_request'] = True
                plc.write([('Vision_Pass', self.state['io_out']['PASS']), ('Vision_Fail', self.state['io_out']['FAIL'])])
        except: pass

    def perform_inspection(self, img_input):
        if self.model is None: 
            self.state['result_status'] = "ERROR: NO YOLO MODEL FOUND"
            return
        
        # YOLO native full-frame inference
        results = self.model(img_input, verbose=False)[0]
        
        # --- NEW: Draw the bounding boxes onto the image ---
        annotated_frame = results.plot()
        cv2.imwrite("temp_capture.jpg", annotated_frame)
        self.state['last_captured_frame'] = True # Freeze HMI on this image
        
        is_pass = False
        highest_conf = 0.0
        best_class = -1
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf > highest_conf:
                highest_conf = conf
                best_class = cls_id
                
        if best_class != -1:
            cfg = self.state['class_configs'][best_class]
            is_pass = highest_conf >= cfg['threshold']
            self.state['result_status'] = f"FOUND {cfg['name']}: {'PASS' if is_pass else 'FAIL'} {highest_conf:.1%}"
        else:
            self.state['result_status'] = "NO OBJECT DETECTED"

        self.state['io_out']['PASS'] = is_pass
        self.state['io_out']['FAIL'] = not is_pass
        self.state['history'].append(f"[{time.strftime('%H:%M:%S')}] {self.state['result_status']}")