import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained DCGAN generator model
try:
    generator = load_model('generator.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the class labels
class_labels = ['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Ahegao']
num_classes = len(class_labels)
latent_dim = 100  # Latent dimension size used for the GAN model

def generate_images(generator, class_idx, num_images=5):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    labels = np.full((num_images, 1), class_idx)
    generated_images = generator.predict([noise, labels])
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    return generated_images

# Streamlit UI
st.title("DCGAN Model for Face Expression Generation")

# Create a select box for class selection
selected_class = st.selectbox("Select Expression Class", options=class_labels)

if st.button('Generate Images'):
    class_idx = class_labels.index(selected_class)
    images = generate_images(generator, class_idx)
    
    if len(images) > 0:
        st.write(f'Generated images for {selected_class} expression:')
        
        # Display images
        fig, ax = plt.subplots(1, len(images), figsize=(15, 15))
        for i, img in enumerate(images):
            img = img.reshape(64, 64, 3)  # Reshape image
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            ax[i].imshow(img)
            ax[i].axis('off')
        st.pyplot(fig)
