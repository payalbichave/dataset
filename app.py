import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Configurations
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'plant_disease_cnn_model.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Get class names exactly as they were during training
TRAIN_DIR = 'train_val_test/train'
if os.path.exists(TRAIN_DIR):
    class_names = sorted(os.listdir(TRAIN_DIR))
else:
    class_names = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
        'Tomato_healthy'
    ]

def model_predict(img_path, model):
    # Depending on how the model was trained, resize the image to 64x64
    img = image.load_img(img_path, target_size=(64, 64))
    # Convert image to array
    img_array = image.img_to_array(img)
    # The model has Rescaling(1./255) as its first layer so we don't scale it here
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    return preds

def format_class_name(name):
    # Clean up the dataset folder names to make them readable
    name = name.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')
    return name

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No image selected for uploading'})
    
    if f and model is not None:
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        try:
            # Make prediction
            preds = model_predict(file_path, model)
            predicted_class_idx = np.argmax(preds, axis=1)[0]
            confidence = np.max(preds) * 100
            
            original_class = class_names[predicted_class_idx]
            readable_class = format_class_name(original_class)
            
            is_healthy = 'healthy' in original_class.lower()

            result = {
                'class': readable_class,
                'confidence': float(confidence),
                'is_healthy': is_healthy,
                'error': None
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Model not loaded correctly on the server.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
