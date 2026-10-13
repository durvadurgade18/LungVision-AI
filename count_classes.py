from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from collections import Counter

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    "DATASET/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

counts = Counter(train_generator.classes)

print("\nTraining Images Per Class:\n")

for class_name, class_index in train_generator.class_indices.items():
    print(f"{class_name}: {counts[class_index]}")