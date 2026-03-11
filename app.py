from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import json

app = Flask(__name__)

# ✅ Load trained model (USE .keras version)
model = tf.keras.models.load_model("plant_disease_cnn.keras")

# ✅ Load class names
with open("class_indices.json", "r") as f:
    class_names = json.load(f)

# ✅ Disease Information Dictionary (keep yours here)
disease_info = {

"Pepper__bell___Bacterial_spot": {
    "symptoms": [
        "Small dark water-soaked spots on leaves.",
        "Spots turn brown with yellow halo.",
        "Leaves may drop prematurely.",
        "Raised scabby lesions on fruits."
    ],
    "remedy": [
        "Use certified disease-free seeds.",
        "Apply copper-based bactericides.",
        "Avoid overhead irrigation.",
        "Practice crop rotation."
    ]
},

"Pepper__bell___healthy": {
    "symptoms": [
        "Leaves are green and firm.",
        "No visible spots or discoloration.",
        "Normal plant growth.",
        "Healthy fruit development."
    ],
    "remedy": [
        "Maintain proper irrigation schedule.",
        "Apply balanced fertilizers.",
        "Monitor regularly for pests.",
        "Ensure proper sunlight exposure."
    ]
},

"Potato___Early_blight": {
    "symptoms": [
        "Small dark brown spots on older leaves.",
        "Concentric ring pattern on lesions.",
        "Yellowing around infected areas.",
        "Premature leaf drop."
    ],
    "remedy": [
        "Apply protective fungicides.",
        "Maintain proper spacing between plants.",
        "Avoid excessive moisture.",
        "Remove infected plant debris."
    ]
},

"Potato___Late_blight": {
    "symptoms": [
        "Dark water-soaked lesions on leaves.",
        "White fungal growth under leaves.",
        "Rapid plant wilting.",
        "Dark patches on tubers."
    ],
    "remedy": [
        "Spray copper-based fungicides.",
        "Improve field drainage.",
        "Remove infected plants immediately.",
        "Use resistant varieties."
    ]
},

"Potato___healthy": {
    "symptoms": [
        "Uniform green foliage.",
        "No visible lesions or spots.",
        "Strong stem growth.",
        "Healthy tuber formation."
    ],
    "remedy": [
        "Maintain proper watering.",
        "Use balanced fertilizers.",
        "Monitor for early disease signs.",
        "Ensure good soil drainage."
    ]
},

"Tomato_Bacterial_spot": {
    "symptoms": [
        "Small dark spots on leaves.",
        "Spots may have yellow halos.",
        "Leaf curling and drop.",
        "Raised lesions on fruits."
    ],
    "remedy": [
        "Use disease-free seeds.",
        "Apply copper sprays.",
        "Avoid working in wet conditions.",
        "Practice crop rotation."
    ]
},

"Tomato_Early_blight": {
    "symptoms": [
        "Brown circular spots with rings.",
        "Yellowing around lesions.",
        "Lower leaves dry and fall.",
        "Dark spots on stems."
    ],
    "remedy": [
        "Remove infected leaves.",
        "Apply fungicides like mancozeb.",
        "Avoid overhead watering.",
        "Rotate crops yearly."
    ]
},

"Tomato_Late_blight": {
    "symptoms": [
        "Large dark greasy spots on leaves.",
        "White fungal growth in humidity.",
        "Rapid plant collapse.",
        "Brown lesions on fruits."
    ],
    "remedy": [
        "Remove infected plants.",
        "Apply systemic fungicides.",
        "Reduce excess irrigation.",
        "Improve air circulation."
    ]
},

"Tomato_Leaf_Mold": {
    "symptoms": [
        "Yellow patches on upper leaves.",
        "Olive-green mold underneath leaves.",
        "Leaf curling.",
        "Reduced fruit production."
    ],
    "remedy": [
        "Reduce greenhouse humidity.",
        "Increase air flow.",
        "Remove infected leaves.",
        "Apply fungicide if severe."
    ]
},

"Tomato_Septoria_leaf_spot": {
    "symptoms": [
        "Small circular gray spots.",
        "Dark brown borders around spots.",
        "Leaves turn yellow.",
        "Lower leaves drop first."
    ],
    "remedy": [
        "Remove infected leaves.",
        "Apply appropriate fungicide.",
        "Avoid wetting foliage.",
        "Use crop rotation."
    ]
},

"Tomato_Spider_mites_Two_spotted_spider_mite": {
    "symptoms": [
        "Tiny yellow specks on leaves.",
        "Fine webbing under leaves.",
        "Leaves turn bronze or dry.",
        "Reduced plant vigor."
    ],
    "remedy": [
        "Spray water to remove mites.",
        "Use insecticidal soap.",
        "Introduce natural predators.",
        "Maintain proper humidity."
    ]
},

"Tomato__Target_Spot": {
    "symptoms": [
        "Circular spots with concentric rings.",
        "Dark brown lesions.",
        "Leaf yellowing.",
        "Fruit spots in severe cases."
    ],
    "remedy": [
        "Apply fungicide sprays.",
        "Remove infected debris.",
        "Improve air circulation.",
        "Avoid overcrowding plants."
    ]
},

"Tomato__Tomato_YellowLeaf__Curl_Virus": {
    "symptoms": [
        "Yellowing of leaf edges.",
        "Upward curling of leaves.",
        "Stunted plant growth.",
        "Reduced fruit yield."
    ],
    "remedy": [
        "Control whitefly population.",
        "Remove infected plants.",
        "Use resistant varieties.",
        "Apply insecticides if needed."
    ]
},

"Tomato__Tomato_mosaic_virus": {
    "symptoms": [
        "Mottled light and dark green patches.",
        "Leaf distortion.",
        "Stunted growth.",
        "Reduced fruit quality."
    ],
    "remedy": [
        "Remove infected plants.",
        "Disinfect tools regularly.",
        "Avoid tobacco contact.",
        "Use resistant seeds."
    ]
},

"Tomato_healthy": {
    "symptoms": [
        "Bright green leaves.",
        "No visible spots.",
        "Strong stem growth.",
        "Healthy fruit formation."
    ],
    "remedy": [
        "Maintain proper irrigation.",
        "Provide balanced nutrients.",
        "Regular monitoring.",
        "Ensure adequate sunlight."
    ]
}

}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html")

    # ✅ Read uploaded image correctly
    img_bytes = BytesIO(file.read())

    img = image.load_img(img_bytes, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ❌ DO NOT divide by 255 here
    # Normalization is already inside the trained model

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    predicted_class = class_names[np.argmax(prediction)]

    print("RAW PREDICTION:", prediction)
    print("PREDICTED:", predicted_class)
    print("CONFIDENCE:", confidence)

    info = disease_info.get(predicted_class)

    return render_template("index.html", result={
        "disease": predicted_class,
        "symptoms": info["symptoms"],
        "remedy": info["remedy"],
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)