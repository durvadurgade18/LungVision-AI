import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ===============================
# Dataset Paths
# ===============================

TRAIN_DIR = "XRAY_VALIDATOR/train"
VAL_DIR = "XRAY_VALIDATOR/val"
TEST_DIR = "XRAY_VALIDATOR/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# ===============================
# Data Generators
# ===============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Classes:", train_generator.class_indices)

# ===============================
# Build Model
# ===============================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# Callbacks
# ===============================

os.makedirs("models", exist_ok=True)

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/xray_validator.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )

]

# ===============================
# Training
# ===============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)


# ===============================
# Evaluation
# ===============================

test_loss, test_accuracy = model.evaluate(test_generator)


print("\n===============================")
print("X-Ray Validator Results")
print("===============================")

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# ===============================
# Save Final Model
# ===============================

model.save("models/xray_validator_final.keras")

print("\nModel Saved Successfully!")