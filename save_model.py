# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:26:35 2023

@author: eyalw
"""

import onnxmltools
from keras.models import load_model

# Load your LSTM model
model = load_model('model.h5')

# Convert the model to ONNX format
onnx_model = onnxmltools.convert_keras(model, target_opset=12)

# Save the ONNX model
onnxmltools.utils.save_model(onnx_model, 'fraud_detection_lstm.onnx')