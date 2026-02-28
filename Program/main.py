import multiprocessing as mp
import os, json
from vision_engine import VisionEngine
from hmi_app import HMIApp

def start_system():
    # Setup Directories
    for folder in ['models', 'config'] + [f'dataset/Class_{i}' for i in range(5)]:
        os.makedirs(folder, exist_ok=True)

    # Load Persistent Settings
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
                defaults.update(json.load(f))
        except: pass

    manager = mp.Manager()
    state = manager.dict({
        'mode': 'RUN', 'trigger_request': False, 'last_captured_frame': False,
        'result_status': "READY", 'system_message': "Ready", 'history': manager.list(),
        'reload_request': True, 'cam_reload': True,
        'active_program': defaults['active_program'],
        'program_list': manager.list(defaults['program_list']),
        'io_mode': defaults['io_mode'], 'cam_source': defaults['cam_source'],
        'basler_ip': defaults['basler_ip'], 'plc_ip': defaults['plc_ip'],
        'class_configs': manager.dict({int(k): v for k, v in defaults['class_configs'].items()}),
        'io_in': manager.dict({'TRIGGER': False, 'MODE_SEL': False}),
        'io_out': manager.dict({'PASS': False, 'FAIL': False, 'RUNNING': False})
    })

    # Initialize Engine (Heavy objects like Flask init inside run_loop)
    engine = VisionEngine(state)
    p = mp.Process(target=engine.run_loop)
    p.daemon = True
    p.start()

    # Run HMI in Main Thread
    HMIApp(state).run()

if __name__ == "__main__":
    start_system()