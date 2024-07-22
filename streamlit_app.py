import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# Define the Autoencoder class
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
            nn.Linear(1000, 100),  # Assuming dim_z is 100
            nn.BatchNorm1d(100),
            nn.ReLU()
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

# Load the VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_auto = Autoencoder().to(device)

try:
    state_dict = torch.load('autoencoder_model.pth', map_location=device)
    model_auto.load_state_dict(state_dict, strict=False)
    model_auto.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to generate images for a given class
def generate_images(model, class_label, num_images=5):
    try:
        # Generate random latent vectors
        z = np.random.randn(num_images, 100)  # Assuming dim_z is 100
        z = torch.FloatTensor(z).to(device)
        
        with torch.no_grad():
            generated_images = model.decode(z).cpu()
        
        return generated_images
    except Exception as e:
        st.error(f"Error generating images: {e}")
        return []

# Streamlit app
st.title('Face Expression Generator')

# Select class
class_label = st.selectbox('Select Expression', ['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Ahegao'])

# Generate and display images
if st.button('Generate Images'):
    images = generate_images(model_auto, class_label)
    if len(images) > 0:
        st.write(f'Generated images for {class_label} expression:')
        
        # Display images
        fig, ax = plt.subplots(1, len(images), figsize=(15, 15))
        for i, img in enumerate(images):
            img = img.view(64, 64, 3)  # Reshape image
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            ax[i].imshow(img.numpy())
            ax[i].axis('off')
        st.pyplot(fig)