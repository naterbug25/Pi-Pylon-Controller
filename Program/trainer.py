import tensorflow as tf
import os

def train_local_model(model_name, num_classes=5):
    """Trains a model locally using Transfer Learning on MobileNetV2"""
    base_dir = "dataset"
    img_size = (224, 224)
    batch_size = 32

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, validation_split=0.2, subset="training", seed=123,
            image_size=img_size, batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir, validation_split=0.2, subset="validation", seed=123,
            image_size=img_size, batch_size=batch_size
        )

        base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=5)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open(f"models/{model_name}.tflite", 'wb') as f:
            f.write(tflite_model)
        return True
    except Exception as e:
        print(f"Training Error: {e}")
        return False