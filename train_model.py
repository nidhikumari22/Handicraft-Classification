import tensorflow as tf
from tensorflow.keras import layers, models
import json

# =====================
# CONFIGURATION
# =====================
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8         # SMALL batch for small dataset
EPOCHS = 30             # Enough with early stopping
SEED = 42

# =====================
# LOAD DATASET
# =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# =====================
# PERFORMANCE OPTIMIZATION
# =====================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =====================
# DATA AUGMENTATION (STRONG – FOR SMALL DATA)
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =====================
# BUILD EFFICIENTNET MODEL
# =====================
def build_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False   # IMPORTANT

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =====================
# TRAIN MODEL
# =====================
model = build_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "efficientnet_best_handicraft_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


import matplotlib.pyplot as plt

# Get accuracy values
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

epochs_range = range(1, len(train_acc) + 1)

# Create plot
plt.figure()
plt.plot(epochs_range, train_acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()

# ✅ SAVE GRAPH AS JPG
plt.savefig("accuracy_vs_epoch.jpg", dpi=300, bbox_inches="tight")

# Show graph
plt.show()

# =====================
# SAVE FINAL MODEL
# =====================
print("✅ Best model saved as efficientnet_best_handicraft_model.keras")
