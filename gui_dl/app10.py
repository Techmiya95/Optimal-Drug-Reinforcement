from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Define mse as a function for compatibility
def mse(y_true, y_pred):
    mse_obj = MeanSquaredError()
    return mse_obj(y_true, y_pred)

# Load model with the custom object
model = load_model(r'E:\gui_dl\rl_model.h5', custom_objects={"mse": mse})


 # Pass the custom function here


# Load model and scaler
try:
    dataset_path = r'E:\gui_dl\drug_dosage_optimization_dataset.csv'

    # Load the trained TensorFlow model

    # Load the dataset and fit the scaler
    data = pd.read_csv(dataset_path)
    scaler = MinMaxScaler().fit(data[[
        "tumor_cell_count",
        "immune_cell_count",
        "desired_tumor_cell_count",
        "desired_immune_cell_count",
        "uncertainty_factor"
    ]])
except Exception as e:
    print(f"Error loading resources: {e}")
    model, scaler = None, None

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if scaler is None or model is None:
        return render_template('index.html', prediction_text="Error: Resources not initialized.")

    try:
        # Extract input data from the form
        input_data = [
            float(request.form.get(key, 0)) for key in [
                "tumor_cell_count",
                "immune_cell_count",
                "desired_tumor_cell_count",
                "desired_immune_cell_count",
                "uncertainty_factor"
            ]
        ]

        # Scale the input data
        input_data_scaled = scaler.transform([input_data])

        # Predict the optimal drug dosage
        prediction = model.predict(input_data_scaled)
        dosage = prediction[0][0]

        # Render the result
        return render_template('index.html', prediction_text=f"Optimal Drug Dosage: {dosage:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
