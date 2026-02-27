import cv2, time, os, numpy as np, platform, threading
from pycomm3 import LogixDriver
from flask import Flask, Response

try:
    from pypylon import pylon
    BASLER_SUPPORT = True
except ImportError:
    BASLER_SUPPORT = False

IS_PI = platform.system() != "Windows"
if IS_PI: from periphery import GPIO

class VisionEngine:
    def __init__(self, state):
        self.state = state
        self.net = None
        self.camera = None

    def gen_frames(self):
        """Generator for Web View stream"""
        while True:
            if os.path.exists("live_buffer.jpg"):
                with open("live_buffer.jpg", "rb") as f:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n')
            time.sleep(0.05)

    def init_camera(self):
        if self.camera: 
            try: self.camera.release() 
            except: self.camera.Close()

        if self.state['cam_source'] == 'Basler' and BASLER_SUPPORT:
            try:
                info = pylon.DeviceInfo()
                info.SetIpAddress(self.state['basler_ip'])
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
                self.camera.Open()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except: self.camera = None
        else:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def grab_frame(self):
        if not self.camera: return None
        if self.state['cam_source'] == 'Basler':
            res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if res.GrabSucceeded():
                img = res.Array
                res.Release()
                return img
            return None
        ret, frame = self.camera.read()
        return frame if ret else None

    def run_loop(self):
        """Main process loop. Flask and Camera are initialized here to avoid pickle errors."""
        # 1. Start Web Server local to this process
        web_app = Flask(__name__)
        @web_app.route('/')
        def video_feed():
            return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        threading.Thread(target=lambda: web_app.run(host='0.0.0.0', port=5000), daemon=True).start()

        # 2. Initialize Hardware
        self.init_camera()
        self.load_resources()

        # 3. Execution Loop
        while True:
            if self.state.get('cam_reload'):
                self.init_camera(); self.state['cam_reload'] = False
            
            if self.state['io_mode'] == 'PLC': self.sync_plc()
            
            if self.state.get('reload_request'):
                self.load_resources(); self.state['reload_request'] = False

            frame = self.grab_frame()
            if frame is None: continue

            cv2.imwrite("live_buffer.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            if self.state.get('trigger_request'):
                self.state['trigger_request'] = False
                if self.state['mode'] == 'RUN': self.perform_inspection(frame)
                else: cv2.imwrite("temp_capture.jpg", frame); self.state['last_captured_frame'] = True
            time.sleep(0.01)

    def load_resources(self):
        p = f"models/{self.state['active_program']}.tflite"
        if os.path.exists(p):
            self.net = cv2.dnn.readNetFromTFLite(p)

    def sync_plc(self):
        try:
            with LogixDriver(self.state['plc_ip']) as plc:
                trig = plc.read('Vision_Trigger')
                if trig.value: self.state['trigger_request'] = True
                plc.write([('Vision_Pass', self.state['io_out']['PASS']), ('Vision_Fail', self.state['io_out']['FAIL'])])
        except: pass

    def perform_inspection(self, frame):
        if self.net is None: return
        self.state['io_out']['RUNNING'] = True
        blob = cv2.dnn.blobFromImage(frame, 1.0/127.5, (224,224), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob); out = self.net.forward()
        pred = int(np.argmax(out[0])); score = float(out[0][pred])
        cfg = self.state['class_configs'][pred]; is_pass = score >= cfg['threshold']
        
        self.state['io_out']['PASS'] = is_pass
        self.state['io_out']['FAIL'] = not is_pass
        self.state['result_status'] = f"{cfg['name']}: {'PASS' if is_pass else 'FAIL'} {score:.0%}"
        
        self.state['history'].append(f"{time.strftime('%H:%M:%S')} - {cfg['name']} Result")
        time.sleep(0.1); self.state['io_out']['RUNNING'] = False