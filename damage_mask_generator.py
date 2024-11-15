import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the model
model = load_model(r"C:\Users\Niharika Kashyap\Pictures\satellite_segmentation_model.h5")

def preprocess_image(image):
    image_resized = cv2.resize(image, (256, 256)) / 255.0
    return np.expand_dims(image_resized, axis=0)

def predict_mask(image):
    preprocessed_image = preprocess_image(image)
    predicted_mask = model.predict(preprocessed_image)[0, :, :, 0]
    return predicted_mask

def show_image_with_mask(original_image, mask):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show the original image
    ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Show the predicted mask with a color map
    cax = ax[1].imshow(mask, cmap='jet')
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    
    # Add a color bar to represent intensity
    fig.colorbar(cax, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04)
    
    # Add a legend for color labels
    ax[2].axis("off")
    ax[2].text(0.5, 0.8, "Legend:", fontsize=14, ha='center', fontweight='bold')
    ax[2].text(0.5, 0.6, "Red/Yellow: Undamaged Area", color="Red", fontsize=12, ha='center')
    ax[2].text(0.5, 0.4, "Light Blue: Minor Damaged Area", color="blue", fontsize=12, ha='center')
    ax[2].text(0.5, 0.2, "Blue: Damaged Area", color="blue", fontsize=12, ha='center')
    
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit UI
st.title("Satellite Image Segmentation")
st.write("Upload a satellite image to see its segmentation mask with a heat bar and color labels.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Make predictions
    mask = predict_mask(image)
    
    # Display results
    show_image_with_mask(image, mask)