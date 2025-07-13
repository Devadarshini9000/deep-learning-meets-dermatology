import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('Skin_disease_model.h5')

# Define class names corresponding to your model
class_names = ['ACNE', 'ATOPIC', 'BCC']  # Replace with actual class names

# Streamlit app title
st.title("Automated Skin Disease Detection Using Deep Learning")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize the image to match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to [0,1] range
    return img_array

# If an image is uploaded, display it and make a prediction
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction
    predictions = model.predict(img_array)
    confidence_scores = predictions[0]
    predicted_class = np.argmax(confidence_scores)
    predicted_class_name = class_names[predicted_class]

    # Display confidence scores for each class
    st.write("### Confidence scores for each class:")
    for i, score in enumerate(confidence_scores):
        st.write(f"- **{class_names[i]}**: {score:.4f}")

    # Highlight the confidence for the Atopic class
    st.write("\n### Atopic Class Analysis:")
    st.write(f"- Confidence for **Atopic**: **{confidence_scores[1]:.4f}**")
    if confidence_scores[1] > 0.5:
        st.write("🔍 **The model seems confident about the Atopic class.**")
    else:
        st.write(" Low confidence for Atopic class. ")

    # Display the predicted class with its confidence
    st.write("\n### Final Prediction:")
    st.write(f"- **Predicted Class**: **{predicted_class_name}**")
    st.write(f"- **Confidence**: **{confidence_scores[predicted_class]:.4f}**")

    # Add a warning if the model's confidence for the predicted class is low
    confidence_threshold = 0.5
    if confidence_scores[predicted_class] < confidence_threshold:
        st.write(
            "The model's confidence for this prediction is low."
        )
