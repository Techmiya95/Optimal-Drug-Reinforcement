from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load model and scaler
model = tf.keras.models.load_model('E:\Techmiya_Machine_Learning_Project\Optimial Drug Dosage Reinforcement learning\gui_dl\rl_model.h5')
data = pd.read_csv('dataset/drug_dosage_optimization_dataset.csv')
scaler = MinMaxScaler().fit(data[["tumor_cell_count", "immune_cell_count",
                                  "desired_tumor_cell_count", "desired_immune_cell_count",
                                  "uncertainty_factor"]])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return render_template('index.html', prediction_text=f"Optimal Drug Dosage: {prediction[0][0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
