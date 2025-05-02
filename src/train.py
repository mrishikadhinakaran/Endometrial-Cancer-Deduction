import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0

# ✅ Ensure dataset path is correct
DATASET_PATH = "dataset/"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset path '{DATASET_PATH}' does not exist.")

# ✅ Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_DIR = "models/"
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "im4MEC_model.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "im4MEC_model.h5")

# ✅ Ensure 'models/' directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔄 Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% for validation
)

# 📂 Load Data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),  # Using 'train' since validation split is applied
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# 🏗️ Load Pretrained Model (EfficientNetB0)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# 🔬 Model Architecture
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),  # L2 Regularization
    layers.Dropout(0.5),  # Dropout
    layers.Dense(train_generator.num_classes, activation="softmax")
])

# 🏆 Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 📉 Learning Rate Scheduler
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# 🚀 Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler]
)

# 💾 Save Model
model.save(MODEL_PATH_KERAS)
model.save(MODEL_PATH_H5)

print(f"Model saved successfully at {MODEL_PATH_KERAS}")
print(f"Model saved successfully at {MODEL_PATH_H5}")
