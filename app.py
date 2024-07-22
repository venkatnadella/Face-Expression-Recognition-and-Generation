from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import os

#functions to integrate predictions
def preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def make_prediction(file_path, model):
    image = preprocess_image(file_path)
    images = np.array(image) / 255.0
    probs = model.predict(images)
    classes = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
    probs_list = probs.tolist()[0]
    predicted_class = classes[probs_list.index(max(probs_list))]
    return predicted_class

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/generate')
def generate():
    return render_template('generate.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
        model_choice = request.form['model']
        
        if model_choice == 'cnn':
            model_name = 'CNN'
            model = load_model('tunedCNNmodel.h5')
        elif model_choice == 'vgg_add_layers':
            model_name = 'Transfer Learnt Model on VGG by Adding Layers'
            model = load_model('tunedTL-addingLayersModel.h5')
        elif model_choice == 'vgg_unfreeze_layers':
            model_name = 'Transfer Learnt Model on VGG by Unfreezing Layers'
            model = load_model('tunedTL-unfreezingLayersModel.h5')
        else:
            return 'Something went wrong'

        prediction = make_prediction(file_path,model)
        
        return render_template('prediction.html', model_choice=model_name, prediction=prediction, filename=file.filename)
    return redirect(request.url)

@app.route('/generate_face', methods=['POST'])
def generate_face():
    generation_choice = request.form['generation_model']

    if model_choice == 'dcgan':
        return 'Angry'
    elif model_choice == 'vae':
        return 'Happy'
    else:
        return 'Something went wrong'
    return f"Generated with {generation_choice}"

if __name__ == '__main__':
    app.run(debug=True)
