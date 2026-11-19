import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to processed dataset
data_dir = r"D:\DOMAIN PROJECT\OUTPUT"

# Create folder to save model
save_dir = r"D:\DOMAIN PROJECT\MODEL"
os.makedirs(save_dir, exist_ok=True)

# Image size and batch size
img_size = (224, 224)
batch_size = 32

# Create train/validation data generators
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,   # increase to 10+ for better accuracy
    validation_data=val_data
)


# Save the trained model (final fix)
model_path = r"D:\DOMAIN PROJECT\project code\eggplant_cnn_model.keras"
model.save(model_path)
print("✅ Model saved successfully at:", model_path)

