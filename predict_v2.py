# ==========================================================
# LUNG DISEASE PREDICTION - VERSION 2
# EfficientNetB0 Final Prediction Pipeline
# ==========================================================

import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

MODEL_PATH = "models/best_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")

print("Model output classes:", model.output_shape[-1])


# ==========================================================
# CLASS LABELS
# MUST MATCH TRAINING ORDER
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
# PREDICT IMAGE
# ==========================================================

# ==========================================================
# ASK USER FOR X-RAY IMAGE
# ==========================================================

IMAGE_PATH = input("\nEnter X-ray image path: ")


# Check file exists
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found. Check the path.")
    exit()


# Load image
img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)


# Convert image to array
img_array = image.img_to_array(img)


# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)


# EfficientNet preprocessing
img_array = preprocess_input(img_array)


# Predict
prediction = model.predict(img_array)


# Get result
predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = np.max(prediction) * 100





# Display result
print("\n================================")
print("   LUNG DISEASE PREDICTION")
print("================================")

print("Predicted Disease :", predicted_class)
print(f"Confidence        : {confidence:.2f}%")

print("\nClass Probabilities:")
for i, prob in enumerate(prediction[0]):
    print(f"{class_names[i]} : {prob*100:.2f}%")


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_xray(image_path):

    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return


    # Load image
    img = image.load_img(
        image_path,
        target_size=(224,224),
        color_mode="rgb"
    )


    # Convert to array
    img_array = image.img_to_array(img)


    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)


    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)


    # Prediction
    prediction = model.predict(img_array)


    probabilities = prediction[0]


    # Highest probability class
    predicted_index = np.argmax(probabilities)

    predicted_class = class_names[predicted_index]

    confidence = probabilities[predicted_index] * 100



    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    print("\n==========================================")
    print("      LUNG DISEASE PREDICTION RESULT")
    print("==========================================")

    print("\nImage:")
    print(os.path.basename(image_path))


    print("\nPredicted Disease:")
    print(predicted_class)


    print("\nConfidence:")
    print(f"{confidence:.2f}%")


    print("\n------------------------------------------")
    print("Top 3 Predictions")
    print("------------------------------------------")


    top3 = np.argsort(probabilities)[::-1][:3]


    for i, index in enumerate(top3):

        print(
            f"{i+1}. {class_names[index]} : "
            f"{probabilities[index]*100:.2f}%"
        )


    print("------------------------------------------")


    if confidence >= 90:
        print("Status: ✅ Very High Confidence")

    elif confidence >= 70:
        print("Status: ✅ Good Confidence")

    else:
        print("Status: ⚠ Low Confidence - Review Image")


    print("==========================================")



# ==========================================================
# TEST IMAGE
# Change this name whenever you test another X-ray
# ==========================================================

image_path = "test_xray.jpg"


predict_xray(image_path)