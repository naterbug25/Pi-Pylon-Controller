import os, cv2, sys, time, json, threading, shutil
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from trainer import train_local_model

class AnnotationLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.drawing = False; self.rect = QRect(); self.start_pt = QPoint()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True; self.start_pt = event.pos(); self.rect = QRect(self.start_pt, self.start_pt)
    def mouseMoveEvent(self, event):
        if self.drawing: self.rect = QRect(self.start_pt, event.pos()).normalized(); self.update()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.drawing = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.rect.isNull():
            p = QPainter(self); p.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)); p.drawRect(self.rect)

class SettingsDialog(QDialog):
    def __init__(self, state, parent):
        super().__init__(parent); self.state = state; self.setWindowTitle("Hardware Config")
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Camera:")); self.cam = QComboBox(); self.cam.addItems(["Webcam", "Basler"]); self.cam.setCurrentText(state['cam_source']); lay.addWidget(self.cam)
        lay.addWidget(QLabel("Basler IP:")); self.b_ip = QLineEdit(state['basler_ip']); lay.addWidget(self.b_ip)
        lay.addWidget(QLabel("PLC IP:")); self.p_ip = QLineEdit(state['plc_ip']); lay.addWidget(self.p_ip)
        lay.addWidget(QLabel("IO Mode:")); self.io = QComboBox(); self.io.addItems(["PLC", "GPIO"]); self.io.setCurrentText(state['io_mode']); lay.addWidget(self.io)
        btn = QPushButton("Save"); btn.clicked.connect(self.save); lay.addWidget(btn); self.setLayout(lay)
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
            self.window.setWindowTitle("Pi-Pylon Controller (YOLO Edition)"); self.window.setFixedSize(1260, 950)
            self.window.setStyleSheet("background-color: #121212; color: #eee;")
            central = QWidget(); main_layout = QHBoxLayout()

            # LEFT
            left = QVBoxLayout()
            left.setAlignment(Qt.AlignmentFlag.AlignTop) # <-- NEW: Packs everything to the top

            header = QHBoxLayout()
            self.msg_lbl = QLabel("READY"); self.msg_lbl.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
            self.hb_light = PilotLight("ENGINE", QColor(0, 150, 255)); header.addWidget(self.msg_lbl, 5); header.addWidget(self.hb_light, 1)
            left.addLayout(header)
            
            # <-- CHANGED: AlignHCenter instead of AlignCenter so it doesn't fight the AlignTop command
            self.vid_lbl = AnnotationLabel(); self.vid_lbl.setFixedSize(640, 480); self.vid_lbl.setStyleSheet("border: 2px solid #555;")
            left.addWidget(self.vid_lbl, alignment=Qt.AlignmentFlag.AlignHCenter) 
            self.pbar = QProgressBar(); self.pbar.hide(); left.addWidget(self.pbar)

            btns = QHBoxLayout()
            self.btn_m = QPushButton("MODE"); self.btn_m.clicked.connect(self.toggle_mode)
            self.btn_t = QPushButton("TRIGGER"); self.btn_t.clicked.connect(self.request_trigger)
            self.btn_s = QPushButton("SETTINGS"); self.btn_s.clicked.connect(self.open_settings)
            self.btn_tr = QPushButton("TRAIN YOLO"); self.btn_tr.clicked.connect(self.start_training)
            [btns.addWidget(b) for b in [self.btn_m, self.btn_t, self.btn_s, self.btn_tr]]; left.addLayout(btns)

            io_panel = QHBoxLayout()
            self.lights_in = {k: PilotLight(k) for k in self.state['io_in'].keys()}
            self.lights_out = {k: PilotLight(k) for k in self.state['io_out'].keys()}
            [io_panel.addWidget(l) for l in list(self.lights_in.values()) + list(self.lights_out.values())]
            left.addLayout(io_panel)

            self.teach_box = QGroupBox("Draw Box on Image & Save Annotation"); t_lay = QHBoxLayout()
            self.class_selector = QComboBox()
            for i in range(5): self.class_selector.addItem(f"Class {i}")
            t_lay.addWidget(self.class_selector)
            self.btn_save_ann = QPushButton("Save Annotation"); self.btn_save_ann.clicked.connect(self.save_annotation); t_lay.addWidget(self.btn_save_ann)
            self.teach_box.setLayout(t_lay); left.addWidget(self.teach_box)
            
            left.addStretch() # <-- NEW: Absorbs all remaining empty space at the bottom of the left panel
            main_layout.addLayout(left, 2)

            # RIGHT
            right = QVBoxLayout()
            right.setAlignment(Qt.AlignmentFlag.AlignTop) # <-- NEW: Pack the right side to the top too
            
            right.addWidget(QLabel("CLASSES & THRESHOLDS"))
            self.class_widgets = {}
            for i in range(5):
                r = QHBoxLayout(); n = QLineEdit(self.state['class_configs'][i]['name']); n.editingFinished.connect(self.update_cfg)
                s = QDoubleSpinBox(); s.setRange(0, 1.0); s.setValue(self.state['class_configs'][i]['threshold']); s.valueChanged.connect(self.update_cfg)
                r.addWidget(QLabel(f"#{i}")); r.addWidget(n); r.addWidget(s); right.addLayout(r); self.class_widgets[i] = (n, s)

            # Add a little spacing
            right.addSpacing(20)

            right.addWidget(QLabel("JOB MANAGEMENT"))
            self.prog_list_ui = QListWidget(); self.prog_list_ui.setFixedHeight(120)
            for p in self.state['program_list']: self.prog_list_ui.addItem(p)
            self.prog_list_ui.itemClicked.connect(self.select_program); right.addWidget(self.prog_list_ui)
            p_ctrls = QHBoxLayout(); self.btn_new = QPushButton("NEW JOB"); self.btn_new.clicked.connect(self.add_program)
            self.btn_del = QPushButton("DELETE JOB"); self.btn_del.clicked.connect(self.delete_program)
            p_ctrls.addWidget(self.btn_new); p_ctrls.addWidget(self.btn_del); right.addLayout(p_ctrls)

            right.addSpacing(20)
            
            right.addWidget(QLabel("INSPECTION HISTORY"))
            self.hist = QListWidget(); right.addWidget(self.hist)
            
            main_layout.addLayout(right, 1)
            central.setLayout(main_layout); self.window.setCentralWidget(central); self.update_ui_visibility()
            
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
        d = {'active_program': self.state['active_program'], 'program_list': list(self.state['program_list']), 'class_configs': {str(k): v for k, v in self.state['class_configs'].items()}, 'plc_ip': self.state['plc_ip'], 'basler_ip': self.state['basler_ip'], 'cam_source': self.state['cam_source'], 'io_mode': self.state['io_mode']}
        with open('config/settings.json', 'w') as f: json.dump(d, f, indent=4)
        
    def toggle_mode(self):
        self.state['mode'] = "TRAIN" if self.state['mode'] == "RUN" else "RUN"
        self.state['last_captured_frame'] = False 
        self.vid_lbl.rect = QRect(); self.update_ui_visibility()
        
    def update_ui_visibility(self):
        is_train = self.state['mode'] == "TRAIN"; self.teach_box.setVisible(is_train)
        
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
        
        if not self.vid_lbl.drawing:
            f = "temp_capture.jpg" if self.state.get('last_captured_frame') else "live_buffer.jpg"
            if os.path.exists(f):
                try:
                    img = cv2.imread(f); r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); q = QImage(r.data, r.shape[1], r.shape[0], r.shape[1]*3, QImage.Format.Format_RGB888); self.vid_lbl.setPixmap(QPixmap.fromImage(q))
                except: pass

    def request_trigger(self):
        self.state['last_captured_frame'] = False
        self.state['trigger_request'] = True
        self.vid_lbl.rect = QRect()
        
    def save_annotation(self):
        if not os.path.exists("temp_capture.jpg") or self.vid_lbl.rect.isNull(): return
        
        img_w = 640.0; img_h = 480.0
        rx, ry, rw, rh = self.vid_lbl.rect.x(), self.vid_lbl.rect.y(), self.vid_lbl.rect.width(), self.vid_lbl.rect.height()
        
        x_center = (rx + rw / 2.0) / img_w
        y_center = (ry + rh / 2.0) / img_h
        width = rw / img_w
        height = rh / img_h
        
        cls_id = self.class_selector.currentIndex()
        uid = int(time.time())
        
        shutil.copy("temp_capture.jpg", f"dataset/images/train/{uid}.jpg")
        with open(f"dataset/labels/train/{uid}.txt", "w") as f:
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            
        self.state['last_captured_frame'] = False
        self.vid_lbl.rect = QRect()
        self.state['result_status'] = f"SAVED YOLO ANNOTATION"

    def run(self): self.window.show(); sys.exit(self.app.exec())