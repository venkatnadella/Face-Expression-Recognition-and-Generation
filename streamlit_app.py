import streamlit as st
import numpy as np
import torch
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn


# Define Autoencoder class as before
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64*3, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 64*64*3)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
    
def load_classification_models():
    try:
        # Load your classification models here
        vgg_add_layers = load_model('tunedTL-addingLayersModel.h5')
        vgg_unfreeze_layers = load_model('tunedTL-unfreezingLayersModel.h5')
        cnn_model = load_model('tunedCNNmodel.h5')
        return vgg_add_layers, vgg_unfreeze_layers, cnn_model
    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        return None, None, None

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = Autoencoder().to(device)
try:
    dcgan_model = load_model('generator.h5')
    state_dict = torch.load('autoencoder_model.pth', map_location=device)
    vae_model.load_state_dict(state_dict, strict=False)
    vae_model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

vgg_add_layers, vgg_unfreeze_layers, cnn_model = load_classification_models()

class_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Ahegao']
num_classes = len(class_labels)
latent_dim = 100 

# Define the function to generate images using VAE
def generate_vae_images(model, num_images=5):
    z = np.random.randn(num_images, 100)  # Latent dimension size
    z = torch.FloatTensor(z).to(device)
    with torch.no_grad():
        generated_images = model.decode(z).cpu()
    return generated_images

# Define the function to generate images using DCGAN
def generate_dcgan_images(generator, class_idx, num_images=5):
    latent_dim = 100
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    labels = np.full((num_images, 1), class_idx)
    generated_images = generator.predict([noise, labels])
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    return generated_images

# Define the function to classify images
def classify_image(model, img):
    try:
        # Resize the image to 224x224 pixels
        img_resized = tf.image.resize(img, (224, 224))

        # Ensure the image is in the correct shape (batch_size, height, width, channels)
        img_batch = np.expand_dims(img_resized, axis=0)

        # Convert the image to a writable array
        img_batch = img_batch.copy()

        # Preprocess the image
        img_preprocessed = preprocess_input(img_batch)

        # Make prediction
        prediction = model.predict(img_preprocessed)

        # Decode prediction
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        prediction_confidence = prediction[0][predicted_class_index]

        return predicted_class_label, prediction_confidence

        # Return the prediction
        return prediction
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

# Streamlit app layout
st.title('Face Expression Generation and Classification')

option = st.sidebar.selectbox('Select Mode', ['Generate Images', 'Classify Images'])

if option == 'Generate Images':
    st.subheader('Image Generation')
    model_type = st.radio('Select Model', ['VAE', 'DCGAN'])
    if model_type == 'VAE':
        class_labels = ['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Ahegao']
        selected_class = st.selectbox("Select Expression Class", options=class_labels)
        if st.button('Generate VAE Images'):
            class_idx = class_labels.index(selected_class)
            images = generate_vae_images(vae_model)
            if len(images) > 0:
                st.write(f'Generated images for {selected_class} expression:')
        
                # Display images
                fig, ax = plt.subplots(1, len(images), figsize=(15, 15))
                for i, img in enumerate(images):
                    img = img.view(64, 64, 3)  # Reshape image
                    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                    ax[i].imshow(img.numpy())
                    ax[i].axis('off')
                st.pyplot(fig)
    elif model_type == 'DCGAN':
        class_labels = ['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Ahegao']
        selected_class = st.selectbox("Select Expression Class", options=class_labels)
        if st.button('Generate DCGAN Images'):
            class_idx = class_labels.index(selected_class)
            images = generate_dcgan_images(dcgan_model, class_idx)
            if images is not None:
                fig, ax = plt.subplots(1, len(images), figsize=(15, 15))
                for i, img in enumerate(images):
                    img = img.reshape(64, 64, 3)
                    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
                    ax[i].imshow(img)
                    ax[i].axis('off')
                st.pyplot(fig)

elif option == 'Classify Images':
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file:
        # Load and preprocess the image
        img = keras_image.load_img(uploaded_file, target_size=(64, 64))  # Load image with target size (not needed if resizing later)
        img = keras_image.img_to_array(img)  # Convert image to numpy array

        # Select classification model
        model_option = st.selectbox('Select Classification Model', ['VGG Add Layers', 'VGG Unfreeze Layers', 'CNN'])
        model = None
        if model_option == 'VGG Add Layers':
            model = vgg_add_layers
        elif model_option == 'VGG Unfreeze Layers':
            model = vgg_unfreeze_layers
        elif model_option == 'CNN':
            model = cnn_model

        # Classify image
        if model:
            predicted_class_label, prediction_confidence = classify_image(model, img)
            if predicted_class_label is not None:
                st.write(f"Prediction: {predicted_class_label} with confidence {prediction_confidence:.2f}")