import os, cv2, sys, time, platform, json, threading
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
        central = QWidget(); main_layout = QHBoxLayout(); main_layout.setContentsMargins(5,5,5,5)

        # LEFT
        left = QVBoxLayout(); left.setSpacing(5)
        self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        self.msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); left.addWidget(self.msg_lbl)
        
        self.vid_lbl = QLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;"); left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Training Progress Bar
        self.pbar = QProgressBar(); self.pbar.setStyleSheet("QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #4CAF50; }")
        self.pbar.hide(); left.addWidget(self.pbar)

        btns = QHBoxLayout()
        self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
        self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
        self.btn_tr = QPushButton("TRAIN LOCAL"); self.btn_tr.clicked.connect(self.start_training)
        if IS_WINDOWS: [b.hide() for b in [self.btn_m, self.btn_tr]]
        btns.addWidget(self.btn_m); btns.addWidget(self.btn_t); btns.addWidget(self.btn_tr); left.addLayout(btns)

        self.teach_box = QGroupBox("Capture Samples"); t_lay = QHBoxLayout()
        for i in range(5):
            b = QPushButton(f"Add #{i}"); b.clicked.connect(lambda chk, a=i: self.save_sample(f"Class_{a}"))
            t_lay.addWidget(b)
        self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box)
        main_layout.addLayout(left, 2)

        # RIGHT
        right = QVBoxLayout(); right.addWidget(QLabel("LOGGER"))
        self.hist = QListWidget(); right.addWidget(self.hist); main_layout.addLayout(right, 1)
        central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()

    def toggle_mode(self): 
        self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"
        self.state['result_status'] = "TRAIN MODE: ARMED" if self.state['mode'] == "TRAIN" else "READY"
        self.update_ui_visibility()

    def update_ui_visibility(self): self.teach_box.setVisible(self.state['mode'] == "TRAIN" and not IS_WINDOWS)

    def start_training(self):
        self.pbar.show(); self.pbar.setValue(0)
        threading.Thread(target=self.run_train, daemon=True).start()

    def run_train(self):
        if train_local_model(self.state['active_program'], self.state):
            self.state['result_status'] = "TRAIN SUCCESS"; self.state['reload_request'] = True
        else: self.state['result_status'] = "TRAIN FAILED"

    def refresh(self):
        # Update Status & Progress
        st = self.state.get('result_status', 'READY'); self.msg_lbl.setText(st)
        self.pbar.setValue(self.state.get('training_progress', 0))
        if self.pbar.value() == 100: QTimer.singleShot(3000, self.pbar.hide)
        
        # Update Logger
        if len(self.state['history']) != self.hist.count():
            self.hist.clear(); [self.hist.addItem(e) for e in reversed(list(self.state['history']))]

        # Video
        f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
        if os.path.exists(f):
            img = cv2.imread(f)
            if img is not None:
                r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q = QImage(r.data, r.shape[1], r.shape[0], r.shape[1]*3, QImage.Format.Format_RGB888)
                self.vid_lbl.setPixmap(QPixmap.fromImage(q))

    def request_trigger(self): self.state['trigger_request'] = True
    def save_sample(self, folder):
        os.makedirs(f"dataset/{folder}", exist_ok=True); shutil.copy("temp_capture.jpg", f"dataset/{folder}/{int(time.time())}.jpg")
        self.state['last_captured_frame'] = False; self.state['result_status'] = "TRAIN MODE: ARMED"
    def run(self): self.window.show(); sys.exit(self.app.exec())