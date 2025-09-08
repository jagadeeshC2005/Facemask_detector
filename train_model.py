# train_model_mobilenet.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

INIT_LR = 1e-4   # learning rate
EPOCHS = 20
BS = 32          # batch size
IMG_SIZE = 128   # image size

# Path to dataset (must contain 'with_mask' and 'without_mask' folders)
dataset_dir = "dataset"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BS,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BS,
    class_mode="binary",
    subset="validation"
)

# Load MobileNetV2 base model (without top layer)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

# Add custom head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

# Put together
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Train
print("[INFO] Training model...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BS,
    epochs=EPOCHS
)

# Save model
print("[INFO] Saving mask detector model...")
model.save("mask_detector_model.h5")

# Plot training history
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.legend()
plt.show()