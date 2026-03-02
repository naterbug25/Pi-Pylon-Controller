import os, yaml, shutil
try:
    from ultralytics import YOLO
except:
    pass

def train_local_model(model_name, state, num_classes=5):
    state['training_progress'] = 0
    yaml_path = "dataset/data.yaml"
    names_list = [state['class_configs'][i]['name'] for i in range(num_classes)]
    
    data_yaml = {
        'train': os.path.abspath('dataset/images/train'),
        'val': os.path.abspath('dataset/images/train'),
        'nc': num_classes,
        'names': names_list
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    try:
        model = YOLO('yolov8n.pt') 

        def on_fit_epoch_end(trainer):
            progress = int(((trainer.epoch + 1) / trainer.epochs) * 100)
            state['training_progress'] = progress

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # Start YOLO Training
        abs_project_path = os.path.abspath('models')
        results = model.train(
            data=yaml_path, 
            epochs=10, 
            imgsz=640,
            project=abs_project_path,
            name=model_name,
            exist_ok=True
        )

        # --- FIX: Dynamically hunt for the generated weights file ---
        best_pt = None
        
        # 1. Try to ask YOLO exactly where it saved it
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            best_pt = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
        
        # 2. Fallback search paths based on YOLO version quirks
        fallbacks = [
            f"{abs_project_path}/{model_name}/weights/best.pt",
            f"runs/detect/models/{model_name}/weights/best.pt",
            f"runs/detect/{model_name}/weights/best.pt"
        ]
        
        if not best_pt or not os.path.exists(best_pt):
            for path in fallbacks:
                if os.path.exists(path):
                    best_pt = path
                    break

        # Move the file to the root models folder for the Vision Engine
        if best_pt and os.path.exists(best_pt):
            target_path = f"models/{model_name}.pt"
            shutil.copy(best_pt, target_path)
            print(f"Successfully moved trained model to {target_path}")
        else:
            print("Trainer Error: Could not locate the best.pt weights file.")
            
        state['training_progress'] = 100
        return True
    except Exception as e:
        print(f"YOLO Trainer Error: {e}")
        return False