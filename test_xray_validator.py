import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


# ===============================
# Load X-Ray Validator Model
# ===============================

model = tf.keras.models.load_model(
    "models/xray_validator.keras"
)

print("Validator Model Loaded Successfully!")


# ===============================
# Test Image Path
# ===============================

img_path = input("Enter image path: ")


# ===============================
# Prepare Image
# ===============================

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(img_array)


# ===============================
# Prediction
# ===============================

prediction = model.predict(img_array)


confidence = prediction[0][0]

xray_probability = (1 - confidence) * 100
not_xray_probability = confidence * 100


print("\nPrediction Score:", confidence)


print("\nX-Ray Probability:", round(xray_probability,2), "%")
print("Not X-Ray Probability:", round(not_xray_probability,2), "%")


if confidence > 0.5:
    print("❌ This is NOT a Chest X-Ray")
else:
    print("✅ This is a Chest X-Ray")