import os, cv2, sys, time, platform, json, threading, shutil
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer
from trainer import train_local_model

class SettingsDialog(QDialog):
    def __init__(self, state, parent):
        super().__init__(parent); self.state = state; self.setWindowTitle("System Settings")
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Camera:")); self.cam = QComboBox(); self.cam.addItems(["Webcam", "Basler"]); self.cam.setCurrentText(state['cam_source']); lay.addWidget(self.cam)
        lay.addWidget(QLabel("Basler IP:")); self.b_ip = QLineEdit(state['basler_ip']); lay.addWidget(self.b_ip)
        lay.addWidget(QLabel("PLC IP:")); self.p_ip = QLineEdit(state['plc_ip']); lay.addWidget(self.p_ip)
        lay.addWidget(QLabel("IO Mode:")); self.io = QComboBox(); self.io.addItems(["PLC", "GPIO"]); self.io.setCurrentText(state['io_mode']); lay.addWidget(self.io)
        btn = QPushButton("Save & Restart"); btn.clicked.connect(self.save); lay.addWidget(btn); self.setLayout(lay)
    def save(self):
        self.state['cam_source'] = self.cam.currentText(); self.state['basler_ip'] = self.b_ip.text()
        self.state['plc_ip'] = self.p_ip.text(); self.state['io_mode'] = self.io.currentText(); self.state['cam_reload'] = True
        self.parent().save_settings_to_disk(); self.accept()

class PilotLight(QWidget):
    def __init__(self, label, active_color=QColor(0, 255, 0)):
        super().__init__(); self.label = label; self.active = False; self.color = active_color; self.setFixedSize(140, 35)
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(self.color if self.active else QColor(50, 50, 50)))
        p.drawEllipse(5, 8, 18, 18); p.setPen(QPen(QColor(220, 220, 220))); p.drawText(35, 22, self.label)

class HMIApp:
    def __init__(self, state):
        self.state = state; self.last_hb = 0
        self.app = QApplication(sys.argv); self.window = QMainWindow(); self.setup_ui()
        self.timer = QTimer(); self.timer.timeout.connect(self.refresh); self.timer.start(33)

    def setup_ui(self):
        self.window.setWindowTitle("Pi-Pylon Controller"); self.window.setFixedSize(1260, 950)
        self.window.setStyleSheet("background-color: #121212; color: #eee;")
        central = QWidget(); main_layout = QHBoxLayout()

        # LEFT
        left = QVBoxLayout()
        header = QHBoxLayout()
        self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        self.hb_light = PilotLight("ENGINE", QColor(0, 150, 255)); header.addWidget(self.msg_lbl, 5); header.addWidget(self.hb_light, 1)
        left.addLayout(header)
        
        self.vid_lbl = QLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;"); left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        self.pbar = QProgressBar(); self.pbar.hide(); left.addWidget(self.pbar)

        btns = QHBoxLayout()
        self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
        self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
        self.btn_s = QPushButton("SETTINGS"); self.btn_s.clicked.connect(self.open_settings)
        self.btn_tr = QPushButton("TRAIN"); self.btn_tr.clicked.connect(self.start_training)
        [btns.addWidget(b) for b in [self.btn_m, self.btn_t, self.btn_s, self.btn_tr]]; left.addLayout(btns)

        io_panel = QHBoxLayout()
        self.lights_in = {k: PilotLight(k) for k in self.state['io_in'].keys()}
        self.lights_out = {k: PilotLight(k) for k in self.state['io_out'].keys()}
        [io_panel.addWidget(l) for l in list(self.lights_in.values()) + list(self.lights_out.values())]
        left.addLayout(io_panel)

        self.teach_box = QGroupBox("Collection (Crop ROI)"); t_lay = QHBoxLayout()
        for i in range(5):
            b = QPushButton(f"Add #{i}"); b.clicked.connect(lambda chk, a=i: self.save_sample(f"Class_{a}")); t_lay.addWidget(b)
        self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box); main_layout.addLayout(left, 2)

        # RIGHT
        right = QVBoxLayout()
        self.roi_tabs = QTabWidget()
        s_tab = QWidget(); s_lay = QVBoxLayout(); self.s_sliders = {}
        for l, k in [("X Min", "x_min"), ("X Max", "x_max"), ("Y Min", "y_min"), ("Y Max", "y_max")]:
            r = QHBoxLayout(); s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, 100); s.setValue(int(self.state['search_roi'][k]*100))
            s.valueChanged.connect(self.update_roi); r.addWidget(QLabel(l)); r.addWidget(s); s_lay.addLayout(r); self.s_sliders[k] = s
        self.roi_tabs.addTab(s_tab, "Search ROI")
        
        c_tab = QWidget(); c_lay = QVBoxLayout(); self.c_sliders = {}
        for l, k in [("X Min", "x_min"), ("X Max", "x_max"), ("Y Min", "y_min"), ("Y Max", "y_max")]:
            r = QHBoxLayout(); s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, 100); s.setValue(int(self.state['crop_roi'][k]*100))
            s.valueChanged.connect(self.update_roi); r.addWidget(QLabel(l)); r.addWidget(s); c_lay.addLayout(r); self.c_sliders[k] = s
        self.roi_tabs.addTab(c_tab, "Training Crop")
        right.addWidget(self.roi_tabs)

        right.addWidget(QLabel("CLASSES & THRESHOLDS"))
        self.class_widgets = {}
        for i in range(5):
            r = QHBoxLayout(); n = QLineEdit(self.state['class_configs'][i]['name']); n.editingFinished.connect(self.update_cfg)
            s = QDoubleSpinBox(); s.setRange(0, 1.0); s.setValue(self.state['class_configs'][i]['threshold']); s.valueChanged.connect(self.update_cfg)
            r.addWidget(QLabel(f"#{i}")); r.addWidget(n); r.addWidget(s); right.addLayout(r); self.class_widgets[i] = (n, s)

        right.addWidget(QLabel("JOB MANAGEMENT"))
        self.prog_list_ui = QListWidget(); self.prog_list_ui.setFixedHeight(120)
        for p in self.state['program_list']: self.prog_list_ui.addItem(p)
        self.prog_list_ui.itemClicked.connect(self.select_program); right.addWidget(self.prog_list_ui)
        p_ctrls = QHBoxLayout(); self.btn_new = QPushButton("NEW JOB"); self.btn_new.clicked.connect(self.add_program)
        self.btn_del = QPushButton("DELETE JOB"); self.btn_del.clicked.connect(self.delete_program)
        p_ctrls.addWidget(self.btn_new); p_ctrls.addWidget(self.btn_del); right.addLayout(p_ctrls)

        self.hist = QListWidget(); right.addWidget(self.hist); main_layout.addLayout(right, 1)
        central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()

    def update_roi(self):
        self.state['search_roi'] = {k: s.value()/100.0 for k, s in self.s_sliders.items()}
        self.state['crop_roi'] = {k: s.value()/100.0 for k, s in self.c_sliders.items()}; self.save_settings_to_disk()
    def open_settings(self): SettingsDialog(self.state, self.window).exec()
    def add_program(self):
        name, ok = QInputDialog.getText(self.window, "New Job", "Name:")
        if ok and name: self.state['program_list'].append(name); self.prog_list_ui.addItem(name); self.save_settings_to_disk()
    def delete_program(self):
        curr = self.prog_list_ui.currentItem()
        if curr and curr.text() != "Part_A": self.state['program_list'].remove(curr.text()); self.prog_list_ui.takeItem(self.prog_list_ui.row(curr)); self.save_settings_to_disk()
    def update_cfg(self):
        c = dict(self.state['class_configs'])
        for i in range(5): c[i] = {'name': self.class_widgets[i][0].text(), 'threshold': self.class_widgets[i][1].value()}
        self.state['class_configs'] = c; self.save_settings_to_disk()
    def select_program(self, item): self.state['active_program'] = item.text(); self.state['reload_request'] = True; self.save_settings_to_disk()
    def save_settings_to_disk(self):
        d = {'active_program': self.state['active_program'], 'program_list': list(self.state['program_list']), 'crop_roi': dict(self.state['crop_roi']), 'search_roi': dict(self.state['search_roi']), 'class_configs': {str(k): v for k, v in self.state['class_configs'].items()}, 'plc_ip': self.state['plc_ip'], 'basler_ip': self.state['basler_ip'], 'cam_source': self.state['cam_source'], 'io_mode': self.state['io_mode']}
        with open('config/settings.json', 'w') as f: json.dump(d, f, indent=4)
    def toggle_mode(self):
        self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"; self.update_ui_visibility()
    def update_ui_visibility(self):
        is_train = self.state['mode'] == "TRAIN"; self.teach_box.setVisible(is_train); self.roi_tabs.setCurrentIndex(1 if is_train else 0)
    def start_training(self): self.pbar.show(); self.pbar.setValue(0); threading.Thread(target=self.run_train, daemon=True).start()
    def run_train(self):
        if train_local_model(self.state['active_program'], self.state): self.state['result_status'] = "SUCCESS"; self.state['reload_request'] = True
    def refresh(self):
        cur_hb = self.state.get('heartbeat', 0)
        self.hb_light.active = (cur_hb != self.last_hb); self.last_hb = cur_hb; self.hb_light.update()
        st = self.state.get('result_status', 'READY'); self.msg_lbl.setText(st)
        self.pbar.setValue(self.state.get('training_progress', 0))
        for k, v in self.state['io_in'].items(): self.lights_in[k].active = v; self.lights_in[k].update()
        for k, v in self.state['io_out'].items(): self.lights_out[k].active = v; self.lights_out[k].update()
        if len(self.state['history']) != self.hist.count():
            self.hist.clear(); [self.hist.addItem(QListWidgetItem(e)) for e in reversed(list(self.state['history']))]
            for i in range(self.hist.count()):
                if ">" in self.hist.item(i).text(): self.hist.item(i).setForeground(QColor(160, 160, 160))
        f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
        if os.path.exists(f):
            try:
                img = cv2.imread(f); r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); q = QImage(r.data, r.shape[1], r.shape[0], r.shape[1]*3, QImage.Format.Format_RGB888); self.vid_lbl.setPixmap(QPixmap.fromImage(q))
            except: pass
    def request_trigger(self): self.state['trigger_request'] = True
    def save_sample(self, folder):
        dest = f"dataset/{folder}/{int(time.time())}.jpg"; os.makedirs(f"dataset/{folder}", exist_ok=True)
        if os.path.exists("temp_capture.jpg"): shutil.copy("temp_capture.jpg", dest); self.state['last_captured_frame'] = False
    def run(self): self.window.show(); sys.exit(self.app.exec())