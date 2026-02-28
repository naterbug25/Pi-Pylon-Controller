import os
import tensorflow as tf

class HMIProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, state):
        super().__init__()
        self.state = state
    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / 5) * 100)
        self.state['training_progress'] = progress

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
        return False

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, class_names=active_classes, validation_split=0.2,
            subset="training", seed=123, image_size=img_size, batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, class_names=active_classes, validation_split=0.2,
            subset="validation", seed=123, image_size=img_size, batch_size=batch_size
        )

        base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(224,224,3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(active_classes), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[HMIProgressCallback(state)], verbose=0)

        model.save(f"models/{model_name}.keras")
        state['training_progress'] = 100
        return True
    except Exception as e:
        print(f"Trainer Error: {e}")
        return False