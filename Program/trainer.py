import os
import tensorflow as tf

class HMIProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, state):
        super().__init__()
        self.state = state
        
    def on_epoch_end(self, epoch, logs=None):
        # Update the progress bar on the HMI (5 epochs total = 20% per epoch)
        self.state['training_progress'] = int(((epoch + 1) / 5) * 100)

def train_local_model(model_name, state, num_classes=5):
    base_dir = "dataset"
    img_size = (224, 224)
    batch_size = 32
    state['training_progress'] = 0
    valid_ext = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')
    
    active_classes = []
    for i in range(num_classes):
        cp = os.path.join(base_dir, f"Class_{i}")
        if os.path.exists(cp) and any(f.lower().endswith(valid_ext) for f in os.listdir(cp)): 
            active_classes.append(f"Class_{i}")
            
    if len(active_classes) < 2: 
        print("Trainer Error: Need at least 2 classes with images to train.")
        return False
        
    try:
        # Load the Datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, class_names=active_classes, validation_split=0.2, 
            subset="training", seed=123, image_size=img_size, batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, class_names=active_classes, validation_split=0.2, 
            subset="validation", seed=123, image_size=img_size, batch_size=batch_size
        )

        # --- NEW: Image Augmentation Pipeline ---
        # This distorts the images slightly every time the AI looks at them, 
        # preventing it from memorizing the static background.
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomBrightness(factor=0.15),
        ])

        # Load MobileNetV2 Base (Pre-trained on ImageNet)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights='imagenet'
        )
        base_model.trainable = False

        # Build the final model architecture
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(224, 224, 3)),
            data_augmentation,                      # 1. Distort the image
            base_model,                             # 2. Extract features
            tf.keras.layers.GlobalAveragePooling2D(), # 3. Flatten features
            tf.keras.layers.Dropout(0.2),           # 4. Drop 20% of connections to prevent overfitting
            tf.keras.layers.Dense(len(active_classes), activation='softmax') # 5. Final Prediction
        ])

        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Train for 5 Epochs
        model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=5, 
            callbacks=[HMIProgressCallback(state)], 
            verbose=0
        )
        
        # Save and finish
        model.save(f"models/{model_name}.keras")
        state['training_progress'] = 100
        return True
        
    except Exception as e: 
        print(f"Trainer Error: {e}")
        return False