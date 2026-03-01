import os, cv2, sys, time, platform, json, threading, shutil
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer
from trainer import train_local_model

class HMIApp:
    def __init__(self, state):
        self.state = state
        self.app = QApplication(sys.argv); self.window = QMainWindow(); self.setup_ui()
        self.timer = QTimer(); self.timer.timeout.connect(self.refresh); self.timer.start(33)

    def setup_ui(self):
        self.window.setWindowTitle("Pi-Pylon-Controller - Dual ROI Mode"); self.window.setFixedSize(1240, 950)
        self.window.setStyleSheet("background-color: #121212; color: #eee;")
        central = QWidget(); main_layout = QHBoxLayout()

        # LEFT
        left = QVBoxLayout()
        self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        self.msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); left.addWidget(self.msg_lbl)
        self.vid_lbl = QLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;"); left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        self.pbar = QProgressBar(); self.pbar.hide(); left.addWidget(self.pbar)

        btns = QHBoxLayout()
        self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
        self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
        self.btn_tr = QPushButton("TRAIN LOCAL"); self.btn_tr.clicked.connect(self.start_training)
        btns.addWidget(self.btn_m); btns.addWidget(self.btn_t); btns.addWidget(self.btn_tr); left.addLayout(btns)

        self.teach_box = QGroupBox("Add Crop Capture to Dataset"); t_lay = QHBoxLayout()
        for i in range(5):
            b = QPushButton(f"Add #{i}"); b.clicked.connect(lambda chk, a=i: self.save_sample(f"Class_{a}"))
            t_lay.addWidget(b)
        self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box); main_layout.addLayout(left, 2)

        # RIGHT
        right = QVBoxLayout()
        
        # Dual ROI Tabs
        self.roi_tabs = QTabWidget()
        
        # Search ROI Tab
        search_tab = QWidget(); s_lay = QVBoxLayout(); self.s_sliders = {}
        for l, k in [("X Min", "x_min"), ("X Max", "x_max"), ("Y Min", "y_min"), ("Y Max", "y_max")]:
            r = QHBoxLayout(); s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, 100)
            s.setValue(int(self.state['search_roi'][k]*100)); s.valueChanged.connect(self.update_search_roi)
            r.addWidget(QLabel(l)); r.addWidget(s); s_lay.addLayout(r); self.s_sliders[k] = s
        btn_rs = QPushButton("Reset Search Window"); btn_rs.clicked.connect(self.reset_search_roi); s_lay.addWidget(btn_rs)
        search_tab.setLayout(s_lay); self.roi_tabs.addTab(search_tab, "Search ROI (Yellow)")

        # Crop ROI Tab
        crop_tab = QWidget(); c_lay = QVBoxLayout(); self.c_sliders = {}
        for l, k in [("X Min", "x_min"), ("X Max", "x_max"), ("Y Min", "y_min"), ("Y Max", "y_max")]:
            r = QHBoxLayout(); s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, 100)
            s.setValue(int(self.state['crop_roi'][k]*100)); s.valueChanged.connect(self.update_crop_roi)
            r.addWidget(QLabel(l)); r.addWidget(s); c_lay.addLayout(r); self.c_sliders[k] = s
        btn_rc = QPushButton("Reset Training Crop"); btn_rc.clicked.connect(self.reset_crop_roi); c_lay.addWidget(btn_rc)
        crop_tab.setLayout(c_lay); self.roi_tabs.addTab(crop_tab, "Training Crop (Red)")

        right.addWidget(self.roi_tabs)

        right.addWidget(QLabel("CLASSES & THRESHOLDS"))
        self.class_widgets = {}
        for i in range(5):
            r = QHBoxLayout(); n = QLineEdit(self.state['class_configs'][i]['name']); n.editingFinished.connect(self.update_cfg)
            s = QDoubleSpinBox(); s.setRange(0, 1.0); s.setValue(self.state['class_configs'][i]['threshold']); s.valueChanged.connect(self.update_cfg)
            r.addWidget(QLabel(f"#{i}")); r.addWidget(n); r.addWidget(s); right.addLayout(r); self.class_widgets[i] = (n, s)

        right.addWidget(QLabel("PROGRAM MANAGEMENT"))
        self.prog_list_ui = QListWidget(); self.prog_list_ui.setFixedHeight(120)
        for p in self.state['program_list']: self.prog_list_ui.addItem(p)
        self.prog_list_ui.itemClicked.connect(self.select_program); right.addWidget(self.prog_list_ui)
        self.btn_del_p = QPushButton("DELETE JOB"); self.btn_del_p.clicked.connect(self.delete_program); right.addWidget(self.btn_del_p)

        self.hist = QListWidget(); right.addWidget(self.hist); main_layout.addLayout(right, 1)

        central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()

    def update_search_roi(self): self.state['search_roi'] = {k: s.value()/100.0 for k, s in self.s_sliders.items()}
    def reset_search_roi(self): [self.s_sliders[k].setValue(v) for k, v in [('x_min', 0), ('x_max', 100), ('y_min', 0), ('y_max', 100)]]; self.update_search_roi()
    
    def update_crop_roi(self): self.state['crop_roi'] = {k: s.value()/100.0 for k, s in self.c_sliders.items()}
    def reset_crop_roi(self): [self.c_sliders[k].setValue(v) for k, v in [('x_min', 25), ('x_max', 75), ('y_min', 25), ('y_max', 75)]]; self.update_crop_roi()

    def update_cfg(self):
        c = dict(self.state['class_configs'])
        for i in range(5): c[i] = {'name': self.class_widgets[i][0].text(), 'threshold': self.class_widgets[i][1].value()}
        self.state['class_configs'] = c; self.save_settings_to_disk()

    def select_program(self, item): self.state['active_program'] = item.text(); self.state['reload_request'] = True; self.save_settings_to_disk()

    def delete_program(self):
        curr = self.prog_list_ui.currentItem()
        if curr and curr.text() != "Part_A":
            self.state['program_list'].remove(curr.text()); self.prog_list_ui.takeItem(self.prog_list_ui.row(curr)); self.save_settings_to_disk()

    def save_settings_to_disk(self):
        d = {'active_program': self.state['active_program'], 'program_list': list(self.state['program_list']), 'crop_roi': dict(self.state['crop_roi']), 'search_roi': dict(self.state['search_roi']), 'class_configs': {str(k): v for k, v in self.state['class_configs'].items()}}
        with open('config/settings.json', 'w') as f: json.dump(d, f, indent=4)

    def toggle_mode(self):
        self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"
        self.state['result_status'] = "TRAIN MODE" if self.state['mode'] == "TRAIN" else "READY"; self.update_ui_visibility()

    def update_ui_visibility(self): self.teach_box.setVisible(self.state['mode'] == "TRAIN")

    def start_training(self): self.pbar.show(); self.pbar.setValue(0); threading.Thread(target=self.run_train, daemon=True).start()

    def run_train(self):
        if train_local_model(self.state['active_program'], self.state):
            self.state['result_status'] = "SUCCESS"; self.state['reload_request'] = True
        else: self.state['result_status'] = "FAILED"

    def refresh(self):
        st = self.state.get('result_status', 'READY'); self.msg_lbl.setText(st)
        self.pbar.setValue(self.state.get('training_progress', 0))
        if len(self.state['history']) != self.hist.count():
            self.hist.clear(); [self.hist.addItem(e) for e in reversed(list(self.state['history']))]
        f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
        if os.path.exists(f):
            try:
                img = cv2.imread(f)
                if img is not None:
                    r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); q = QImage(r.data, r.shape[1], r.shape[0], r.shape[1]*3, QImage.Format.Format_RGB888)
                    self.vid_lbl.setPixmap(QPixmap.fromImage(q))
            except: pass

    def request_trigger(self): self.state['trigger_request'] = True
    def save_sample(self, folder):
        dest = f"dataset/{folder}/{int(time.time())}.jpg"; os.makedirs(f"dataset/{folder}", exist_ok=True)
        if os.path.exists("temp_capture.jpg"): shutil.copy("temp_capture.jpg", dest); self.state['last_captured_frame'] = False
    def run(self): self.window.show(); sys.exit(self.app.exec())