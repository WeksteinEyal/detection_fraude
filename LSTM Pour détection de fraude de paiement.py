# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:11:16 2023

@author: assui
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv(r'C:\Travail\card_transaction_2.csv')
df = df.dropna(subset=['Merchant State'])

# Data preprocessing
df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
df['Time'] = df['Time'].apply(lambda x: int(x[:2])*60+int(x[3:]))
df['Use Chip'] = df['Use Chip'].map({"Chip Transaction": 0, "Online Transaction": 1, "Swipe Transaction": 2})

label_encoder = LabelEncoder()
df['Merchant Name'] = df['Merchant Name'].astype("string")
df['Merchant Name'] = label_encoder.fit_transform(df['Merchant Name'])
df['Merchant City'] = label_encoder.fit_transform(df['Merchant City'])
df['Merchant State'].fillna("ONLINE", inplace=True)
df['Merchant State'] = label_encoder.fit_transform(df['Merchant State'])
df = df.drop('Zip', axis=1)
df['Errors?'].fillna("None", inplace=True)
df['Errors?'] = label_encoder.fit_transform(df['Errors?'])

# Splitting data into training and testing sets
X, y = df.drop('Is Fraud?', axis=1), df['Is Fraud?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create model, required for KerasClassifier
def create_model(units=50, activation='relu', dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False, activation=activation))  # Last LSTM layer
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Wrap Keras model in a KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=1)

# Define grid search parameters
param_grid = {
    'units': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30]
}

# Create GridSearchCV and fit
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train_scaled, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluate using best estimator
best_model = grid_result.best_estimator_
predictions = best_model.predict(X_test_scaled)
binary_predictions = (predictions > 0.5).astype(int)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, binary_predictions)
print(f'Accuracy: {accuracy}')
report = classification_report(y_test, binary_predictions)
print(report)
