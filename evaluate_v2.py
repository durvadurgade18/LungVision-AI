import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/best_model.keras"

TEST_DIR = "DATASET/test"


# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")


# ==========================================================
# CLASS NAMES
# SAME ORDER AS TRAINING
# ==========================================================

class_names = [
    "Atelectasis",
    "COVID19",
    "Emphysema",
    "Lung_Cancer",
    "Normal",
    "Pulmonary_Edema",
    "Pulmonary_Fibrosis",
    "Tuberculosis"
]


# ==========================================================
# TEST DATA GENERATOR
# ==========================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    classes=class_names
)


print("\nClass Mapping:")
print(test_generator.class_indices)


# ==========================================================
# PREDICTION
# ==========================================================

print("\nRunning predictions...")

predictions = model.predict(test_generator)


y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes


# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n================================")
print(" MODEL EVALUATION RESULTS")
print("================================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)