import os, cv2, sys, time, platform, json, threading
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer
from trainer import train_local_model

IS_WINDOWS = platform.system() == "Windows"

class SettingsDialog(QDialog):
    def __init__(self, state, parent):
        super().__init__(parent); self.state = state
        self.setWindowTitle("System Settings"); self.setFixedWidth(350)
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Camera Source:")); self.cam = QComboBox(); self.cam.addItems(["Webcam", "Basler"]); self.cam.setCurrentText(state['cam_source']); lay.addWidget(self.cam)
        lay.addWidget(QLabel("Basler IP:")); self.ip = QLineEdit(state['basler_ip']); lay.addWidget(self.ip)
        lay.addWidget(QLabel("PLC IP:")); self.pip = QLineEdit(state['plc_ip']); lay.addWidget(self.pip)
        lay.addWidget(QLabel("IO Mode:")); self.io = QComboBox(); self.io.addItems(["PLC", "GPIO"]); self.io.setCurrentText(state['io_mode'])
        if IS_WINDOWS: self.io.setEnabled(False)
        lay.addWidget(self.io); btn = QPushButton("Save"); btn.clicked.connect(self.save); lay.addWidget(btn); self.setLayout(lay)

    def save(self):
        self.state['cam_source'] = self.cam.currentText(); self.state['basler_ip'] = self.ip.text()
        self.state['plc_ip'] = self.pip.text(); self.state['io_mode'] = self.io.currentText(); self.state['cam_reload'] = True
        self.parent().save_settings_to_disk(); self.accept()

class PilotLight(QWidget):
    def __init__(self, label):
        super().__init__(); self.label = label; self.active = False; self.setFixedSize(140, 35)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(0,255,0) if self.active else QColor(50,50,50)))
        p.drawEllipse(5, 8, 18, 18); p.setPen(QPen(QColor(220,220,220))); p.drawText(35, 22, self.label)

class HMIApp:
    def __init__(self, state):
        self.state = state
        self.app = QApplication(sys.argv); self.window = QMainWindow(); self.setup_ui()
        self.timer = QTimer(); self.timer.timeout.connect(self.refresh); self.timer.start(33)

    def setup_ui(self):
        self.window.setWindowTitle("Pi-Pylon-Controller"); self.window.setFixedSize(1200, 850)
        self.window.setStyleSheet("background-color: #121212; color: #eee;")
        central = QWidget(); main_layout = QHBoxLayout(); main_layout.setContentsMargins(5,5,5,5)

        left = QVBoxLayout(); left.setSpacing(5)
        self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        self.msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); left.addWidget(self.msg_lbl)
        self.vid_lbl = QLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;"); left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btns = QHBoxLayout()
        self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
        if IS_WINDOWS: self.btn_m.hide() # Hide MODE button on Windows
        
        self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
        self.btn_s = QPushButton("SETTINGS"); self.btn_s.clicked.connect(self.open_settings)
        self.btn_tr = QPushButton("TRAIN LOCAL"); self.btn_tr.clicked.connect(self.start_training)
        if IS_WINDOWS: self.btn_tr.hide() # Hide training on Windows
        
        btns.addWidget(self.btn_m); btns.addWidget(self.btn_t); btns.addWidget(self.btn_s); btns.addWidget(self.btn_tr); left.addLayout(btns)

        io_panel = QHBoxLayout()
        in_box = QGroupBox("IN"); in_lay = QVBoxLayout(); self.lights_in = {k: PilotLight(k) for k in self.state['io_in'].keys()}
        for l in self.lights_in.values(): in_lay.addWidget(l)
        in_box.setLayout(in_lay); io_panel.addWidget(in_box)
        out_box = QGroupBox("OUT"); out_lay = QVBoxLayout(); self.lights_out = {k: PilotLight(k) for k in self.state['io_out'].keys()}
        for l in self.lights_out.values(): out_lay.addWidget(l)
        out_box.setLayout(out_lay); io_panel.addWidget(out_box); left.addLayout(io_panel)

        self.teach_box = QGroupBox("Capture Samples")
        t_lay = QHBoxLayout()
        for i in range(5):
            b = QPushButton(f"Add #{i}"); b.clicked.connect(lambda chk, a=i: self.save_sample(f"Class_{a}"))
            t_lay.addWidget(b)
        self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box); main_layout.addLayout(left, 2)

        right = QVBoxLayout(); right.setSpacing(5)
        right.addWidget(QLabel("CLASSES & THRESHOLDS"))
        self.class_widgets = {}
        for i in range(5):
            r = QHBoxLayout(); n = QLineEdit(self.state['class_configs'][i]['name']); n.editingFinished.connect(self.update_cfg)
            s = QDoubleSpinBox(); s.setRange(0, 1.0); s.setValue(self.state['class_configs'][i]['threshold']); s.valueChanged.connect(self.update_cfg)
            r.addWidget(QLabel(f"#{i}")); r.addWidget(n); r.addWidget(s); right.addLayout(r); self.class_widgets[i] = (n, s)

        right.addWidget(QLabel("PROGRAMS"))
        self.progs = QListWidget(); self.progs.itemClicked.connect(self.sel_prog); self.progs.setFixedHeight(100)
        for p in self.state['program_list']: self.progs.addItem(p)
        right.addWidget(self.progs); self.hist = QListWidget(); right.addWidget(self.hist); main_layout.addLayout(right, 1)
        central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()

    def start_training(self):
        self.state['system_message'] = "TRAINING..."; threading.Thread(target=self.run_train, daemon=True).start()

    def run_train(self):
        if train_local_model(self.state['active_program']):
            self.state['system_message'] = "SUCCESS"; self.state['reload_request'] = True
        else: self.state['system_message'] = "FAILED"

    def open_settings(self): SettingsDialog(self.state, self.window).exec()
    def save_settings_to_disk(self):
        d = {'active_program': self.state['active_program'], 'program_list': list(self.state['program_list']), 'io_mode': self.state['io_mode'], 'cam_source': self.state['cam_source'], 'basler_ip': self.state['basler_ip'], 'plc_ip': self.state['plc_ip'], 'class_configs': {str(k): v for k, v in self.state['class_configs'].items()}}
        with open('config/settings.json', 'w') as f: json.dump(d, f, indent=4)

    def update_cfg(self):
        c = dict(self.state['class_configs'])
        for i in range(5): c[i] = {'name': self.class_widgets[i][0].text(), 'threshold': self.class_widgets[i][1].value()}
        self.state['class_configs'] = c; self.save_settings_to_disk()

    def sel_prog(self, it): self.state['active_program'] = it.text(); self.state['reload_request'] = True; self.save_settings_to_disk()
    def request_trigger(self): self.state['trigger_request'] = True
    def toggle_mode(self): self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"; self.update_ui_visibility()
    def update_ui_visibility(self): 
        if self.state['mode'] == "RUN" or IS_WINDOWS: self.teach_box.hide() # Ensure hidden on Windows
        else: self.teach_box.show()

    def refresh(self):
        st = self.state.get('result_status', 'READY'); self.msg_lbl.setText(st.split(":")[0])
        self.msg_lbl.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {'#4CAF50' if 'PASS' in st or 'READY' in st else '#F44336'};")
        for k, v in self.state['io_in'].items(): self.lights_in[k].active = v; self.lights_in[k].update()
        for k, v in self.state['io_out'].items(): self.lights_out[k].active = v; self.lights_out[k].update()
        if len(self.state['history']) != self.hist.count():
            self.hist.clear(); [self.hist.addItem(e) for e in reversed(list(self.state['history']))]
        f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
        if os.path.exists(f):
            img = cv2.imread(f)
            if img is not None:
                r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w, ch = r.shape
                q = QImage(r.data, w, h, ch*w, QImage.Format.Format_RGB888); self.vid_lbl.setPixmap(QPixmap.fromImage(q))

    def save_sample(self, folder):
        os.makedirs(f"dataset/{folder}", exist_ok=True); cv2.imwrite(f"dataset/{folder}/{int(time.time())}.jpg", cv2.imread("temp_capture.jpg")); self.state['last_captured_frame'] = False
    def run(self): self.window.show(); sys.exit(self.app.exec())