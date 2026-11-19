from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"D:\DOMAIN PROJECT\project code\eggplant_cnn_model.keras")

# Load and preprocess the test image
img_path = r"D:\DOMAIN PROJECT\test_image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"D:\DOMAIN PROJECT\project code\eggplant_cnn_model.keras")

# Load and preprocess the test image
img_path = r"D:\DOMAIN PROJECT\test_image.jpg"  # Change this path for other test images
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Class labels (must match your training order)
class_names = [
    'Aphids', 'Cercospora Leaf Spot', 'Defect Eggplant', 'Flea Beetles',
    'Fresh Eggplant', 'Fresh Eggplant Leaf', 'Leaf Wilt',
    'Phytophthora Blight', 'Powdery Mildew', 'Tobacco Mosaic Virus'
]

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

# Determine if diseased or not
fresh_classes = ['Fresh Eggplant', 'Fresh Eggplant Leaf']

if predicted_class in fresh_classes:
    status = "🟢 No disease detected — Eggplant is healthy."
else:
    status = f"🔴 Diseased plant detected — {predicted_class}."

# Print results
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print(status)

# Show image with label
plt.imshow(image.load_img(img_path))
plt.title(f"{status}\n({confidence:.2f}% Confidence)")
plt.axis("off")
plt.show()

img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Class labels (update if your dataset has different class names)
class_names = [
    'Aphids', 'Cercospora Leaf Spot' , 'Defect Eggplant'  , 'Flea Beetles' , 'Fresh Eggplant' , 
    'Fresh Eggplant Leaf' , 'Leaf Wilt', 'Phytophthora Blight' , 'Powdery Mildew', 'Tobacco Mosaic Virus'
]

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

# Print results
print(f"🩺 Predicted Disease: {predicted_class}")
print(f"🔹 Confidence: {confidence:.2f}%")

# Show image with label
plt.imshow(image.load_img(img_path))
plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()
