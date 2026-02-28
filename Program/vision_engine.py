import cv2, time, os, numpy as np, platform, threading
import tensorflow as tf
from pycomm3 import LogixDriver
from flask import Flask, Response

try:
    from pypylon import pylon
    BASLER_SUPPORT = True
except ImportError:
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
                info = pylon.DeviceInfo()
                info.SetIpAddress(self.state['basler_ip'])
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
                self.camera.Open()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except: self.camera = None
        else:
            self.camera = cv2.VideoCapture(0)

    def grab_frame(self):
        if not self.camera: return None
        if self.state['cam_source'] == 'Basler':
            try:
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    img = res.Array
                    res.Release()
                    return img
            except: return None
        ret, frame = self.camera.read()
        return frame if ret else None

    def run_loop(self):
        web_app = Flask(__name__)
        @web_app.route('/')
        def video_feed():
            return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        threading.Thread(target=lambda: web_app.run(host='0.0.0.0', port=5000), daemon=True).start()

        self.init_camera()
        self.load_resources()

        while True:
            if self.state.get('cam_reload'):
                self.init_camera(); self.state['cam_reload'] = False
            
            if self.state['io_mode'] == 'PLC': self.sync_plc()
            
            if self.state.get('reload_request'):
                self.load_resources(); self.state['reload_request'] = False

            frame = self.grab_frame()
            if frame is None:
                time.sleep(0.1); continue

            cv2.imwrite("live_buffer.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            if self.state.get('trigger_request'):
                self.state['trigger_request'] = False
                if self.state['mode'] == 'RUN': self.perform_inspection(frame)
                else:
                    cv2.imwrite("temp_capture.jpg", frame)
                    self.state['last_captured_frame'] = True
            time.sleep(0.01)

    def load_resources(self):
        p = f"models/{self.state['active_program']}.keras"
        if os.path.exists(p):
            try:
                self.net = tf.keras.models.load_model(p)
                print(f"LOADED: Native Keras Model {p}")
            except Exception as e:
                print(f"Model Load Error: {e}")
                self.net = None

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
            
            # Preprocessing
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Inference
            predictions = self.net.predict(img_array, verbose=0)[0]
            pred_index = int(np.argmax(predictions))
            winner_score = float(predictions[pred_index])
            winner_cfg = self.state['class_configs'][pred_index]
            is_pass = winner_score >= winner_cfg['threshold']

            # --- FORMAT RESULTS FOR HMI ---
            # 1. Main display string (Winner Only)
            main_status = f"{winner_cfg['name']}: {'PASS' if is_pass else 'FAIL'} {winner_score:.1%}"
            
            # 2. Detailed logger string (All Classes)
            details = []
            for i, prob in enumerate(predictions):
                c_name = self.state['class_configs'][i]['name']
                details.append(f"{c_name}: {prob:.0%}")
            all_scores_str = " | ".join(details)

            # Update Shared State
            self.state['result_status'] = main_status # For the 32px Top Label
            self.state['io_out']['PASS'] = is_pass
            self.state['io_out']['FAIL'] = not is_pass
            
            # Log the full breakdown to the History list (Logger Box)
            timestamp = time.strftime('%H:%M:%S')
            self.state['history'].append(f"[{timestamp}] {main_status}")
            self.state['history'].append(f"   > {all_scores_str}") # Indented details
            
            time.sleep(0.1)
            self.state['io_out']['RUNNING'] = False