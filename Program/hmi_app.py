import os, cv2, sys, time, platform, json, threading, shutil
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer
from trainer import train_local_model

IS_WINDOWS = platform.system() == "Windows"

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
        central = QWidget(); main_layout = QHBoxLayout(); main_layout.setContentsMargins(10,10,10,10)

        # LEFT COLUMN
        left = QVBoxLayout(); left.setSpacing(10)
        self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        self.msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); left.addWidget(self.msg_lbl)
        
        self.vid_lbl = QLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;"); left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.pbar = QProgressBar(); self.pbar.setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; color: white; } QProgressBar::chunk { background-color: #4CAF50; }")
        self.pbar.hide(); left.addWidget(self.pbar)

        btns = QHBoxLayout()
        self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
        self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
        self.btn_tr = QPushButton("TRAIN LOCAL"); self.btn_tr.clicked.connect(self.start_training)
        if IS_WINDOWS: [b.hide() for b in [self.btn_m, self.btn_tr]]
        btns.addWidget(self.btn_m); btns.addWidget(self.btn_t); btns.addWidget(self.btn_tr); left.addLayout(btns)

        # IO PANEL
        io_panel = QHBoxLayout()
        in_box = QGroupBox("Input Status"); in_lay = QVBoxLayout(); self.lights_in = {k: PilotLight(k) for k in self.state['io_in'].keys()}
        for l in self.lights_in.values(): in_lay.addWidget(l)
        in_box.setLayout(in_lay); io_panel.addWidget(in_box)
        
        out_box = QGroupBox("Output Status"); out_lay = QVBoxLayout(); self.lights_out = {k: PilotLight(k) for k in self.state['io_out'].keys()}
        for l in self.lights_out.values(): out_lay.addWidget(l)
        out_box.setLayout(out_lay); io_panel.addWidget(out_box)
        left.addLayout(io_panel)

        self.teach_box = QGroupBox("Capture Samples"); t_lay = QHBoxLayout()
        for i in range(5):
            b = QPushButton(f"Add #{i}"); b.clicked.connect(lambda chk, a=i: self.save_sample(f"Class_{a}"))
            t_lay.addWidget(b)
        self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box)
        main_layout.addLayout(left, 2)

        # RIGHT COLUMN
        right = QVBoxLayout(); right.setSpacing(10)
        right.addWidget(QLabel("PROGRAM MANAGEMENT"))
        self.prog_list_ui = QListWidget(); self.prog_list_ui.setFixedHeight(120)
        for p in self.state['program_list']: self.prog_list_ui.addItem(p)
        self.prog_list_ui.itemClicked.connect(self.select_program)
        right.addWidget(self.prog_list_ui)

        p_btns = QHBoxLayout()
        self.btn_add_p = QPushButton("NEW JOB"); self.btn_add_p.clicked.connect(self.add_program)
        self.btn_del_p = QPushButton("DELETE JOB"); self.btn_del_p.clicked.connect(self.delete_program)
        p_btns.addWidget(self.btn_add_p); p_btns.addWidget(self.btn_del_p); right.addLayout(p_btns)

        right.addWidget(QLabel("INSPECTION LOG"))
        self.hist = QListWidget(); right.addWidget(self.hist)
        main_layout.addLayout(right, 1)

        central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()

    def add_program(self):
        name, ok = QInputDialog.getText(self.window, "New Job", "Enter Program Name:")
        if ok and name:
            if name not in self.state['program_list']:
                self.state['program_list'].append(name); self.prog_list_ui.addItem(name)
                self.save_settings_to_disk()

    def delete_program(self):
        curr = self.prog_list_ui.currentItem()
        if curr:
            name = curr.text(); confirm = QMessageBox.question(self.window, "Delete", f"Delete {name}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                self.state['program_list'].remove(name); self.prog_list_ui.takeItem(self.prog_list_ui.row(curr))
                if self.state['active_program'] == name: self.state['active_program'] = "Part_A"
                self.save_settings_to_disk()

    def select_program(self, item):
        self.state['active_program'] = item.text(); self.state['reload_request'] = True; self.save_settings_to_disk()

    def save_settings_to_disk(self):
        d = {'active_program': self.state['active_program'], 'program_list': list(self.state['program_list']), 'plc_ip': self.state['plc_ip'], 'cam_source': self.state['cam_source']}
        with open('config/settings.json', 'w') as f: json.dump(d, f, indent=4)

    def toggle_mode(self):
        self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"
        self.state['result_status'] = "TRAIN MODE: ARMED" if self.state['mode'] == "TRAIN" else "READY"
        self.update_ui_visibility()

    def update_ui_visibility(self): self.teach_box.setVisible(self.state['mode'] == "TRAIN" and not IS_WINDOWS)

    def start_training(self):
        self.pbar.show(); self.pbar.setValue(0); threading.Thread(target=self.run_train, daemon=True).start()

    def run_train(self):
        if train_local_model(self.state['active_program'], self.state):
            self.state['result_status'] = "SUCCESS"; self.state['reload_request'] = True
        else: self.state['result_status'] = "FAILED"

    def refresh(self):
        st = self.state.get('result_status', 'READY'); self.msg_lbl.setText(st)
        color = '#4CAF50' if any(x in st for x in ['PASS', 'READY', 'SUCCESS']) else '#F44336'
        if "ARMED" in st: color = '#FFC107'
        self.msg_lbl.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color};")
        
        self.pbar.setValue(self.state.get('training_progress', 0))
        if self.pbar.value() == 100: QTimer.singleShot(3000, self.pbar.hide)

        # Update IO Pilot Lights
        for k, v in self.state['io_in'].items(): self.lights_in[k].active = v; self.lights_in[k].update()
        for k, v in self.state['io_out'].items(): self.lights_out[k].active = v; self.lights_out[k].update()

        if len(self.state['history']) != self.hist.count():
            self.hist.clear(); [self.hist.addItem(e) for e in reversed(list(self.state['history']))]

        f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
        if os.path.exists(f):
            img = cv2.imread(f)
            if img is not None:
                r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); q = QImage(r.data, r.shape[1], r.shape[0], r.shape[1]*3, QImage.Format.Format_RGB888)
                self.vid_lbl.setPixmap(QPixmap.fromImage(q))

    def request_trigger(self): self.state['trigger_request'] = True
    def save_sample(self, folder):
        dest = f"dataset/{folder}/{int(time.time())}.jpg"; os.makedirs(f"dataset/{folder}", exist_ok=True)
        if os.path.exists("temp_capture.jpg"):
            shutil.copy("temp_capture.jpg", dest); self.state['last_captured_frame'] = False
            self.state['result_status'] = "TRAIN MODE: ARMED"

    def run(self): self.window.show(); sys.exit(self.app.exec())