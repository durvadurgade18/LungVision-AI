import tensorflow as tf

model = tf.keras.models.load_model("models/best_model.keras", compile=False)

print("\n--- ALL LAYERS ---\n")
for layer in model.layers:
    print(f"{layer.name:30s} {layer.__class__.__name__}")

print("\n--- LIKELY CONV LAYERS (for Grad-CAM) ---\n")
for layer in model.layers:
    if "conv" in layer.name.lower() or "Conv" in layer.__class__.__name__:
        print(layer.name)