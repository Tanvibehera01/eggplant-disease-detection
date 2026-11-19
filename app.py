
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Eggplant Disease Detection",
    page_icon="🍆",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #151714;
    color: #F5F5F5;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    color: #55BD60;
    font-size: 40px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #A9B0A8;
    font-size: 19px;
    margin-bottom: 30px;
}

.result-box {
    background-color: #202820;
    border: 1px solid #4C7950;
    border-radius: 12px;
    padding: 20px;
}

div.stButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: #55BD60;
    color: #55BD60;
}

[data-testid="stFileUploader"] {
    background-color: #20251F;
    border: 1px solid #39483A;
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        r"D:\DOMAIN PROJECT\project code\eggplant_cnn_model.keras"
    )

    return model


model = load_model()


# ==========================================
# 5. CLASS NAMES
# ==========================================

class_names = [
    'Aphids',
    'Cercospora Leaf Spot',
    'Defect Eggplant',
    'Flea Beetles',
    'Fresh Eggplant',
    'Fresh Eggplant Leaf',
    'Leaf Wilt',
    'Phytophthora Blight',
    'Powdery Mildew',
    'Tobacco Mosaic Virus'
]

healthy_classes = [
    'Fresh Eggplant',
    'Fresh Eggplant Leaf'
]


# ==========================================
# 6. DISEASE SOLUTIONS
# ==========================================

solutions = {

    'Aphids': [
        'Check the undersides of leaves for aphid colonies.',
        'Remove heavily infested leaves when practical.',
        'Use appropriate insect-control methods recommended locally.',
        'Monitor nearby plants for further infestation.'
    ],

    'Cercospora Leaf Spot': [
        'Remove severely affected leaves.',
        'Avoid overhead watering where possible.',
        'Maintain adequate spacing and air circulation.',
        'Consult an agricultural expert about suitable fungicides.'
    ],

    'Defect Eggplant': [
        'Inspect the fruit for physical damage and discoloration.',
        'Check for possible pest attacks or environmental stress.',
        'Remove damaged fruits if they are unsuitable for use.',
        'Monitor the plant for further symptoms.'
    ],

    'Flea Beetles': [
        'Inspect leaves for small holes and feeding damage.',
        'Remove nearby weeds that may host pests.',
        'Monitor the crop regularly.',
        'Use locally recommended pest-control methods if necessary.'
    ],

    'Leaf Wilt': [
        'Check soil moisture and drainage.',
        'Inspect stems and roots for possible damage.',
        'Avoid excessive watering.',
        'Seek agricultural advice if wilting continues.'
    ],

    'Phytophthora Blight': [
        'Improve field drainage and avoid waterlogging.',
        'Remove severely infected plant material.',
        'Avoid moving contaminated soil between plants.',
        'Consult an agricultural expert about disease management.'
    ],

    'Powdery Mildew': [
        'Remove severely affected leaves.',
        'Improve air circulation around plants.',
        'Avoid excessive nitrogen fertilization.',
        'Seek advice about suitable disease-control products.'
    ],

    'Tobacco Mosaic Virus': [
        'Remove and isolate infected plants as advised locally.',
        'Disinfect gardening tools between plants.',
        'Control weeds that may act as virus hosts.',
        'Avoid handling healthy plants immediately after infected plants.'
    ]
}


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_disease(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# ==========================================
# 8. HOME PAGE
# ==========================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🍆 Eggplant Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'AI-powered detection of eggplant crop diseases 🌿'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("## Welcome to the Detection System")

    st.write(
        "This system uses a trained Convolutional Neural Network "
        "(CNN) model to classify eggplant images into different "
        "disease and healthy categories."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📸")
        st.markdown("**Step 1: Upload Image**")
        st.caption("Choose a clear eggplant or leaf image.")

    with col2:
        st.markdown("### 🤖")
        st.markdown("**Step 2: AI Analysis**")
        st.caption("The CNN model analyzes the image.")

    with col3:
        st.markdown("### 📊")
        st.markdown("**Step 3: View Results**")
        st.caption("Get the predicted class and recommendations.")

    st.markdown("---")

    if st.button("🚀 Start Detection", use_container_width=True):
        go_to("detect")


# ==========================================
# 9. IMAGE UPLOAD PAGE
# ==========================================

elif st.session_state.page == "detect":

    st.markdown(
        '<div class="main-title">📸 Upload an Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Upload a clear image of the eggplant or leaf to begin detection.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Select Your Image")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.upload_key}"
    )

    if uploaded_file is not None:

        try:

            image_bytes = uploaded_file.getvalue()

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            st.markdown("### 🖼️ Image Preview")

            st.image(
                image,
                caption="Uploaded Eggplant Image",
                use_container_width=True
            )

            st.write("")

            if st.button(
                "🔍 Analyze Image",
                use_container_width=True
            ):

                with st.spinner("Analyzing image using AI model..."):

                    predicted_class, confidence, scores = predict_disease(image)

                    st.session_state.uploaded_image = image

                    st.session_state.prediction = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "scores": scores
                    }

                go_to("result")

        except Exception as e:

            st.error(f"Unable to process image: {e}")

    st.markdown("---")

    if st.button("⬅️ Back to Home"):
        go_to("home")


# ==========================================
# 10. SEPARATE RESULT PAGE
# ==========================================

elif st.session_state.page == "result":

    st.markdown(
        '<div class="main-title">📊 Detection Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Eggplant disease analysis report'
        '</div>',
        unsafe_allow_html=True
    )

    if (
        st.session_state.uploaded_image is not None
        and st.session_state.prediction is not None
    ):

        image = st.session_state.uploaded_image

        result = st.session_state.prediction

        predicted_class = result["class"]

        confidence = result["confidence"]

        # Uploaded image

        st.markdown("### 🖼️ Analyzed Image")

        st.image(
            image,
            caption="Uploaded Eggplant Image",
            use_container_width=True
        )

        st.markdown("---")

        # Prediction heading

        st.markdown(
            f"## 🩺 Prediction: {predicted_class}"
        )

        # Confidence

        st.markdown(
            f"**Confidence: {confidence * 100:.2f}%**"
        )

        st.progress(
            max(0.0, min(confidence, 1.0))
        )

        # Healthy or Diseased status

        if predicted_class in healthy_classes:

            st.success(
                "✅ The plant appears Healthy! "
                "Continue monitoring regularly."
            )

        elif predicted_class == "Defect Eggplant":

            st.warning(
                f"⚠️ Detected Class: {predicted_class}"
            )

            st.info(
                "This class represents an eggplant fruit defect, "
                "not necessarily an infectious disease."
            )

        else:

            st.error(
                f"⚠️ Detected Disease: {predicted_class}"
            )

        # Suggested Solutions

        if predicted_class in solutions:

            st.markdown("## 💡 Suggested Solutions:")

            for item in solutions[predicted_class]:

                st.markdown(f"- {item}")

        # Other predictions

        st.markdown("---")

        st.markdown("### 📈 Other Model Predictions")

        top_indices = np.argsort(
            result["scores"]
        )[::-1][:3]

        for index in top_indices:

            st.write(
                f"**{class_names[index]}:** "
                f"{result['scores'][index] * 100:.2f}%"
            )

        st.caption(
            "Note: Model predictions are estimates, not confirmed "
            "agricultural diagnoses. Actual results may vary."
        )

        # Bottom navigation

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔁 Detect Another",
                use_container_width=True
            ):

                st.session_state.prediction = None

                st.session_state.uploaded_image = None

                st.session_state.upload_key += 1

                go_to("detect")

        with col2:

            if st.button(
                "🏠 Home",
                use_container_width=True
            ):

                st.session_state.prediction = None

                st.session_state.uploaded_image = None

                go_to("home")

    else:

        st.warning("No detection result found. Please upload an image first.")

        if st.button("📸 Go to Upload Page"):
            go_to("detect")
