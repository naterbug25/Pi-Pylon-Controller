import multiprocessing as mp
import os, json, sys
from vision_engine import VisionEngine
from hmi_app import HMIApp

os.environ["QT_QPA_PLATFORM"] = "xcb"

def start_system():
    # Setup strictly for YOLO (images and labels)
    for folder in ['models', 'config', 'dataset/images/train', 'dataset/labels/train']:
        os.makedirs(folder, exist_ok=True)

    settings_path = 'config/settings.json'
    defaults = {
        'program_list': ["Part_A"], 'active_program': "Part_A",
        'io_mode': 'PLC', 'cam_source': 'Webcam',
        'basler_ip': '192.168.1.50', 'plc_ip': '192.168.1.10',
        'class_configs': {str(i): {'name': f'Class_{i}', 'threshold': 0.85} for i in range(5)}
    }
    
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except: pass

    manager = mp.Manager()
    state = manager.dict({
        'mode': 'RUN', 'trigger_request': False, 'last_captured_frame': False,
        'result_status': "READY", 'training_progress': 0, 'history': manager.list(),
        'reload_request': True, 'cam_reload': True, 'heartbeat': 0,
        'active_program': defaults['active_program'],
        'program_list': manager.list(defaults['program_list']),
        'io_mode': defaults['io_mode'], 'cam_source': defaults['cam_source'],
        'basler_ip': defaults['basler_ip'], 'plc_ip': defaults['plc_ip'],
        'class_configs': manager.dict({int(k): v for k, v in defaults['class_configs'].items()}),
        'io_in': manager.dict({'TRIGGER': False}),
        'io_out': manager.dict({'PASS': False, 'FAIL': False, 'RUNNING': False})
    })

    engine = VisionEngine(state)
    p = mp.Process(target=engine.run_loop, daemon=True)
    p.start()

    try:
        HMIApp(state).run()
    finally:
        if p.is_alive():
            p.terminate()
            p.join()

if __name__ == "__main__":
    start_system()