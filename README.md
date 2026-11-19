# 🍆 Eggplant Disease Detection System

An AI-powered Eggplant Disease Detection System built using **Python, TensorFlow, Convolutional Neural Networks (CNN), and Streamlit**. The application analyzes uploaded eggplant images, identifies the predicted disease or condition, displays prediction confidence, and provides suggested solutions to support crop health monitoring.

## 📌 Project Overview

Eggplant is affected by various diseases and pests that can reduce crop quality and productivity. Identifying these problems at an early stage can help farmers take appropriate preventive measures.

This project uses a **Convolutional Neural Network (CNN)** to classify eggplant images into different healthy, disease-affected, pest-affected, and defect categories.

The system provides a simple and interactive web interface where users can upload an eggplant image and receive a prediction along with its confidence score and suggested solutions.

## 🎯 Objectives

- Develop an AI-based system for eggplant image classification.
- Identify different eggplant diseases, pests, and healthy conditions.
- Display prediction results with confidence scores.
- Provide suggested solutions based on the predicted category.
- Create a user-friendly web application using Streamlit.

## ✨ Key Features

- **Image Upload:** Upload an eggplant image directly through the web application.
- **AI-Based Classification:** Uses a trained CNN model to analyze the uploaded image.
- **10 Classification Categories:** Recognizes different eggplant conditions.
- **Confidence Score:** Displays the model's confidence for its prediction.
- **Suggested Solutions:** Provides recommendations based on the predicted category.
- **Interactive Interface:** Simple and user-friendly Streamlit application.
- **Separate Result Page:** Displays the prediction and analysis after clicking the Analyze Image button.

## 🔬 Classification Categories

The model is trained to classify images into the following 10 categories:

| No. | Category |
|---|---|
| 1 | Aphids |
| 2 | Cercospora Leaf Spot |
| 3 | Defect Eggplant |
| 4 | Flea Beetles |
| 5 | Fresh Eggplant |
| 6 | Fresh Eggplant Leaf |
| 7 | Leaf Wilt |
| 8 | Phytophthora Blight |
| 9 | Powdery Mildew |
| 10 | Tobacco Mosaic Virus |

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and prediction |
| CNN | Image classification |
| NumPy | Numerical operations and image processing |
| Pillow (PIL) | Image loading and resizing |
| Streamlit | Web application development |
| Matplotlib | Image visualization and analysis |
| Git & GitHub | Version control and project hosting |

## ⚙️ Project Workflow

1. **Dataset Collection:** Collect eggplant images belonging to different disease and healthy categories.
2. **Image Preprocessing:** Resize images to 224 × 224 pixels and prepare them for model training.
3. **Data Splitting:** Divide the dataset into training and validation sets.
4. **Model Development:** Build a CNN model for image classification.
5. **Model Training:** Train the model using the prepared image dataset.
6. **Model Saving:** Save the trained model in `.keras` format.
7. **Prediction:** Load the trained model and analyze uploaded images.
8. **Web Application:** Integrate the model with Streamlit to display predictions, confidence scores, and suggested solutions.

## 📂 Project Structure

```text
eggplant-disease-detection/
│
├── app.py
├── DATA INPUT.py
├── IMAGE CLASSIFICATION.py
├── modeltraining.py
├── predicting.py
│
├── eggplant_cnn_model.keras
│
├── README.md
├── .gitignore
└── requirements.txt
```



## 🚀 How to Use

1. Open the Eggplant Disease Detection web application.
2. Navigate to the image detection section.
3. Upload an eggplant image.
4. Click on the **Analyze Image** button.
5. View the uploaded image and predicted category.
6. Check the prediction confidence score.
7. Read the suggested solutions for the predicted condition.
8. Click **Detect Another** to analyze another image.

## 📊 Model Information

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input Image Size:** 224 × 224 pixels
- **Number of Classes:** 10
- **Training and Validation Split:** 80% training and 20% validation
- **Batch Size:** 32
- **Training Epochs:** 10
- **Model Format:** `.keras`



