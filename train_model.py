import tensorflow as tf
from tensorflow.keras import layers, models
import json

DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def build_model(model_name):
    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    if model_name == "custom_cnn":
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)

    elif model_name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3)
        )
        base.trainable = False
        x = base(x)
        x = layers.GlobalAveragePooling2D()(x)

    elif model_name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3)
        )
        base.trainable = False
        x = base(x)
        x = layers.GlobalAveragePooling2D()(x)

    else:
        raise ValueError("Unknown model name")

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model_names = ["custom_cnn", "mobilenet", "efficientnet"]


for name in model_names:
    print(f"\n🚀 Training {name.upper()}...\n")

    model = build_model(name)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    model.save(f"{name}_handicraft_model.h5")
    print(f"Model saved as {name}_handicraft_model.h5")

