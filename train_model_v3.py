# ==========================================================
# LUNG DISEASE PREDICTION USING EFFICIENTNET
# Version 3 - Competition Pipeline
# ==========================================================

import os
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    LearningRateScheduler
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC



# ==========================================================
# GPU SETTINGS
# ==========================================================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU detected and enabled.")

    except RuntimeError as e:
        print(e)

else:
    print("Training on CPU.")

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# PATHS
# ==========================================================

TRAIN_DIR = "DATASET/train"
VAL_DIR = "DATASET/val"
TEST_DIR = "DATASET/test"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================

IMG_SIZE = 224

BATCH_SIZE = 16

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 10

TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

LEARNING_RATE = 1e-3
NUM_CLASSES = 8

# ==========================================================
# IMAGE AUGMENTATION
# ==========================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    brightness_range=[0.85,1.15],

    horizontal_flip=False,

    fill_mode="nearest"

)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# ==========================================================
# DATA GENERATORS
# ==========================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=SEED

)

val_generator = val_test_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

test_generator = val_test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

NUM_CLASSES = train_generator.num_classes


print(f"Number of classes: {NUM_CLASSES}")
# ==========================================================
# CLASS WEIGHTS
# ==========================================================

classes = train_generator.classes

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(classes),

    y=classes

)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")

for k,v in class_weights.items():
    print(k,"=",round(v,3))


# ==========================================================
# CUSTOM FOCAL LOSS
# Compatible with TensorFlow 2.15
# ==========================================================

def focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow((1 - y_pred), gamma)

        loss = weight * cross_entropy

        return tf.reduce_mean(loss)

    return loss


# ==========================================================
# COSINE LEARNING RATE SCHEDULER
# ==========================================================

def cosine_decay(epoch):

    min_lr = 1e-6

    max_lr = LEARNING_RATE

    cosine = 0.5 * (
        1 + np.cos(np.pi * epoch / TOTAL_EPOCHS)
    )

    return min_lr + (max_lr - min_lr) * cosine


# ==========================================================
# LOAD EFFICIENTNETB0
# ==========================================================

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(IMG_SIZE, IMG_SIZE, 3)

)

base_model.trainable = False


# ==========================================================
# CLASSIFICATION HEAD
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.30)(x)

x = Dense(

    512,

    activation="relu"

)(x)

x = BatchNormalization()(x)

x = Dropout(0.20)(x)

predictions = Dense(

    NUM_CLASSES,

    activation="softmax"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=predictions

)


# ==========================================================
# COMPILE MODEL
# ==========================================================

optimizer = Adam(

    learning_rate=LEARNING_RATE

)

model.compile(

    optimizer=optimizer,

    loss=focal_loss(),

    metrics=[

        "accuracy",

        Precision(name="precision"),

        Recall(name="recall"),

        AUC(name="auc")

    ]

)

print("\n")

model.summary()

print("\nModel compiled successfully.")

# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

csv_logger = CSVLogger(
    "training_log.csv",
    append=False
)

lr_scheduler = LearningRateScheduler(
    cosine_decay,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stop,
    csv_logger,
    lr_scheduler
]

# ==========================================================
# STAGE 1 TRAINING (Frozen EfficientNet)
# ==========================================================

print("\n" + "="*60)
print("STAGE 1 : TRAINING CLASSIFIER HEAD")
print("="*60)

history_stage1 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)

# ==========================================================
# STAGE 2 : FINE-TUNING
# ==========================================================

print("\n" + "=" * 60)
print("STAGE 2 : FINE-TUNING EFFICIENTNET")
print("=" * 60)

# Unfreeze the last 30 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

print("\nTrainable Layers:")

trainable_count = 0

for layer in base_model.layers:
    if layer.trainable:
        trainable_count += 1

print(trainable_count)

# Recompile with a lower learning rate
model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss=focal_loss(),

    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]

)

history_stage2 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=TOTAL_EPOCHS,

    initial_epoch=history_stage1.epoch[-1] + 1,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

best_model = tf.keras.models.load_model(
    "models/best_model.keras",
    compile=False
)

best_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

best_model.save("models/final_model.keras")

os.makedirs("graphs", exist_ok=True)

print("\nFinal model saved successfully!")

# ==========================================================
# PLOT TRAINING HISTORY
# ==========================================================

acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]

loss = history_stage1.history["loss"] + history_stage2.history["loss"]
val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]

plt.figure(figsize=(8,6))
plt.plot(acc,label="Train Accuracy")
plt.plot(val_acc,label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("graphs/accuracy_plot.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(loss,label="Train Loss")
plt.plot(val_loss,label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("graphs/loss_plot.png")   # ✅ Correct
plt.show()
# ==========================================================
# EVALUATE BEST MODEL
# ==========================================================

print("\n" + "="*60)
print("EVALUATING BEST MODEL")
print("="*60)

best_model = tf.keras.models.load_model(
    "models/best_model.keras",
    compile=False
)

best_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

results = best_model.evaluate(
    test_generator,
    verbose=1
)

print("\nTest Results")
print("-"*40)

for name,value in zip(best_model.metrics_names,results):
    print(f"{name}: {value:.4f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = best_model.predict(test_generator)

y_pred = np.argmax(predictions,axis=1)

y_true = test_generator.classes

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

print(classification_report(

    y_true,

    y_pred,

    target_names=list(test_generator.class_indices.keys())

))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(8,8))

plt.imshow(cm,cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

classes = list(test_generator.class_indices.keys())

plt.xticks(range(len(classes)),classes,rotation=90)

plt.yticks(range(len(classes)),classes)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")

plt.show()

print("\nTraining Completed Successfully!")

print("Best Model : models/best_model.keras")

print("Final Model : models/final_model.keras")