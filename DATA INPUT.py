
import os
import cv2
import numpy as np

base_dir = r"D:\DOMAIN PROJECT\DATA"

input_dirs = [
    os.path.join(base_dir, "Original dataset"),
    os.path.join(base_dir, "Augmented dataset")
]

output_dir = r"D:\DOMAIN PROJECT\OUTPUT"
target_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

for input_dir in input_dirs:

    for class_folder in os.listdir(input_dir):

        class_input = os.path.join(input_dir, class_folder)
        class_output = os.path.join(output_dir, class_folder)

        os.makedirs(class_output, exist_ok=True)

        for img_name in os.listdir(class_input):

            img_path = os.path.join(class_input, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_resized = cv2.resize(img, target_size)
            cv2.imwrite(
                os.path.join(class_output, img_name),
                img_resized
            )

print("All images processed successfully!")
