from flask import Flask, request, jsonify, render_template
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the trained model
model = tf.keras.models.load_model(
    r"E:\Techmiya_Machine_Learning_Project\Optimial Drug Dosage Reinforcement learning\GUI\dqn_drug_dosage_model.h5",
    custom_objects={"mse": MeanSquaredError()}
)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess input state
    state = np.array([
        0 if data['Gender'] == 'Male' else 1,
        float(data['Age']) / 100,
        float(data['Height']) / 250,
        float(data['Weight']) / 150,
        hash(data['Condition']) % 100 / 100,
        hash(data['Drug']) % 100 / 100,
        0 if data['Comorbidities'] == 'None' else 1,
    ]).reshape(1, -1)

    # Debugging: Log the input state
    print("Input state:", state)

    # Predict action
    action = model.predict(state)[0]

    # Debugging: Log the raw model output
    print("Raw model output:", action)

    # Calculate dosage and clip
    dosage = np.clip(action[0], 0, 1)

    # Debugging: Log the final dosage
    print("Recommended dosage (clipped):", dosage)

    return jsonify({'recommended_dosage': dosage})

if __name__ == '__main__':
    app.run(debug=True)
