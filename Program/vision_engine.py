import cv2, time, os, numpy as np, platform, threading
import tensorflow as tf
from pycomm3 import LogixDriver
from flask import Flask, Response

try:
    from pypylon import pylon
    BASLER_SUPPORT = True
except:
    BASLER_SUPPORT = False

class VisionEngine:
    def __init__(self, state):
        self.state = state
        self.net = None
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

    def grab_frame(self):
        if not self.camera: return None
        if self.state['cam_source'] == 'Basler':
            try:
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    img = res.Array; res.Release(); return img
            except: return None
        ret, frame = self.camera.read()
        return frame if ret else None

    def run_loop(self):
        app = Flask(__name__)
        @app.route('/')
        def video(): return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True).start()

        self.init_camera(); self.load_resources()
        
        while True:
            if self.state.get('cam_reload'): self.init_camera(); self.state['cam_reload'] = False
            if self.state['io_mode'] == 'PLC': self.sync_plc()
            if self.state.get('reload_request'): self.load_resources(); self.state['reload_request'] = False

            frame = self.grab_frame()
            if frame is None: time.sleep(0.1); continue
                # Save to a temporary hidden file first
            cv2.imwrite("live_buffer.tmp.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                # Atomically replace the buffer file (prevents HMI from reading a half-written file)
            os.replace("live_buffer.tmp.jpg", "live_buffer.jpg")

            if self.state.get('trigger_request'):
                self.state['trigger_request'] = False
                if self.state['mode'] == 'RUN':
                    self.perform_inspection(frame)
                else:
                    cv2.imwrite("temp_capture.jpg", frame)
                    self.state['last_captured_frame'] = True
                    self.state['result_status'] = "IMAGE CAPTURED - READY TO STORE"
            time.sleep(0.01)

    def load_resources(self):
        p = f"models/{self.state['active_program']}.keras"
        if os.path.exists(p):
            try: self.net = tf.keras.models.load_model(p)
            except: self.net = None
        else: self.net = None

    def sync_plc(self):
        try:
            with LogixDriver(self.state['plc_ip']) as plc:
                tag_trig = plc.read('Vision_Trigger').value
                self.state['io_in']['TRIGGER'] = tag_trig
                if tag_trig: self.state['trigger_request'] = True
                plc.write([('Vision_Pass', self.state['io_out']['PASS']), ('Vision_Fail', self.state['io_out']['FAIL'])])
        except: pass

    def perform_inspection(self, frame):
        if self.net is None: return
        self.state['io_out']['RUNNING'] = True
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = self.net.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds)); score = float(preds[idx])
        cfg = self.state['class_configs'][idx]; is_pass = score >= cfg['threshold']

        main_txt = f"{cfg['name']}: {'PASS' if is_pass else 'FAIL'} {score:.1%}"
        details = " | ".join([f"{self.state['class_configs'][i]['name']}: {p:.0%}" for i, p in enumerate(preds)])
        
        self.state['result_status'] = main_txt
        self.state['io_out']['PASS'] = is_pass; self.state['io_out']['FAIL'] = not is_pass
        self.state['history'].append(f"[{time.strftime('%H:%M:%S')}] {main_txt}")
        self.state['history'].append(f"   > {details}")
        time.sleep(0.1); self.state['io_out']['RUNNING'] = False