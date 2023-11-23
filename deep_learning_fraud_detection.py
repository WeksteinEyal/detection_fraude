# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:46:31 2023

@author: guery
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from tensorflow.keras.layers import Dense ,SimpleRNN, Dropout, LSTM
from keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import requests
import traceback
TOKEN = 'Put the telegram bot token here'
chat_id = 'put the chat_id'
message = "The process start"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json())

df = pd.read_csv(r"C:\Users\guery\OneDrive\Documents\Fraud detection\card_transaction_2.csv")
#######################################################################################
# Clean Data
#######################################################################################

df = df.loc[df['Merchant City'] == " ONLINE"]
df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
df['Time'] = df['Time'].apply(lambda x: int(x[:2])*60+int(x[3:]))
df['Use Chip'] = df['Use Chip'].map({"Chip Transaction": 0, "Online Transaction": 1, "Swipe Transaction": 2})
label_encoder = LabelEncoder()
df['Merchant Name'] = df['Merchant Name'].astype("string")
df['Merchant Name'] = label_encoder.fit_transform(df['Merchant Name'])
df.drop(['Merchant City', 'Merchant State', 'Zip', 'Month', 'Day', 'Errors?'], axis=1, inplace=True)

df.info()
# Générer la matrice de corrélation
corr = df.corr()

# Afficher la matrice de corrélation avec un dégradé de couleurs
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation')
plt.show()

# Calculer les corrélations avec la variable cible
corr_with_target = corr['Is Fraud?']

# Afficher les corrélations
print("Corrélations avec 'Is Fraud?':")
print(corr_with_target)

plt.figure(figsize=(10, 6))
corr_with_target.plot(kind='bar')
plt.title('Corrélations avec Is Fraud?')
plt.show()


X, y = df.drop('Is Fraud?', axis=1), df['Is Fraud?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for RNN
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

# Create a dictionary to store the results
results = {}

#######################################################################################
# RNN
#######################################################################################
try:
    message = "The RNN process start"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    def create_model_RNN(units=64, units2=64, units3=64, dropout_rate=0.2, activation='sigmoid', dropout_rate2=0.2,
                         activation2='sigmoid', dropout_rate3=0.2, activation3='sigmoid', optimizer='adam'):
        model = Sequential()
        model.add(SimpleRNN(units, input_shape=(1, X_train_scaled.shape[1]), activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(SimpleRNN(units2, activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(SimpleRNN(units3, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    model_RNN = KerasClassifier(build_fn=create_model_RNN, epochs=30, batch_size=64, verbose=1)
    
    param_grid = {
        'units': [8,16,32,64,128],
        'units2': [0,8,16,32,64,128],
        'units3': [0,8,16,32,64,128],
        'dropout_rate': [0.2, 0.3, 0.4],
        "activation": ['relu'],
        "optimizer": ['adam'],
        'batch_size': [32,64],
        'epochs': [10, 50, 100]
    }
    
    grid = GridSearchCV(estimator=model_RNN, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    grid_result = grid.fit(X_train_reshaped, y_train)
    
    # Sauvegarde des resultats
    results['RNN'] = {
        'best_params': grid_result.best_params_,
        'best_accuracy': grid_result.best_score_
    }
    
    best_model = grid_result.best_estimator_
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Make predictions
    predictions = best_model.predict(X_test_reshaped)
    binary_predictions = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, binary_predictions)
    classification_rep = classification_report(y_test, binary_predictions)
    
    results['RNN']['test_accuracy'] = accuracy
    results['RNN']['classification_report'] = classification_rep
    
    message = "The RNN process has completed successfully."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    
    # Format the RNN results for a more readable message
    formatted_rnn_results = (
        f"The RNN results:\n"
        f"Best Parameters: {results['RNN']['best_params']}\n"
        f"Best Accuracy: {results['RNN']['best_accuracy']:.4f}\n"
        f"Test Accuracy: {results['RNN']['test_accuracy']:.4f}\n"
        f"Classification Report:\n{results['RNN']['classification_report']}"
    )
    
    # Send the formatted message
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={formatted_rnn_results}"
    print(requests.get(url).json())
    


except Exception as e:
    # Send a notification if an exception occurs
    error_message = f"An error occurred RNN model: {str(e)}\n\n{traceback.format_exc()}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={error_message}"
    print(requests.get(url).json())
    

#######################################################################################
# LSTM
#######################################################################################
try :
    message = "The LSTM process start"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    
    def create_model_LSTM(units=64,units2=8,units3 = 8, activation='relu', dropout_rate=0.2,optimizer = 'adam'):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units2, return_sequences=True, activation=activation))  
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units3, return_sequences=False, activation=activation))  
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    model_LSTM = KerasClassifier(build_fn=create_model_LSTM, epochs=30, batch_size=64, verbose=1)
    
    param_grid = {
        'units': [8,16,32,64,128],
        'units2': [0,8,16,32,64,128],
        'units3': [0,8,16,32,64,128],
        'dropout_rate': [0.2,0.3],
        "activation": ['relu'],
        "optimizer": ['adam'],
        'batch_size': [32,64],
        'epochs': [50,70]
    }
    grid = GridSearchCV(estimator=model_LSTM, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train_scaled, y_train)
    
    
    results['LSTM'] = {
        'best_params': grid_result.best_params_,
        'best_accuracy': grid_result.best_score_
    }
    
    
    best_model = grid_result.best_estimator_
    predictions = best_model.predict(X_test_scaled)
    binary_predictions = (predictions > 0.5).astype(int)
    
    
    accuracy = accuracy_score(y_test, binary_predictions)
    classification_rep = classification_report(y_test, binary_predictions)
    
    results['LSTM']['test_accuracy'] = accuracy
    results['LSTM']['classification_report'] = classification_rep
    
    message = "The LSTM process has completed successfully."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    
    formatted_lstm_results = (
        f"The LSTM results:\n"
        f"Best Parameters: {results['LSTM']['best_params']}\n"
        f"Best Accuracy: {results['LSTM']['best_accuracy']:.4f}\n"
        f"Test Accuracy: {results['LSTM']['test_accuracy']:.4f}\n"
        f"Classification Report:\n{results['LSTM']['classification_report']}"
    )
    
    # Send the formatted message
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={formatted_lstm_results}"
    print(requests.get(url).json())

except Exception as e:
    # Send a notification if an exception occurs

    error_message = f"An error occurred LSTM model: {str(e)}\n\n{traceback.format_exc()}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={error_message}"
    print(requests.get(url).json())
    
#######################################################################################
# Autoencoder
#######################################################################################

try :
    message = "The Autoencoder process start"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    
    def create_autoencoder_model(units1=64, units2=64,units3=64, dropout_rate = 0.2):
        autoencoder = Sequential()
        autoencoder.add(Dense(units1, input_dim=X_train_resampled.shape[1], activation='relu'))
        autoencoder.add(Dropout(dropout_rate))
        autoencoder.add(Dense(units2, activation='relu'))
        autoencoder.add(Dropout(dropout_rate))
        autoencoder.add(Dense(units3, activation='relu'))
        autoencoder.add(Dense(1, activation='sigmoid'))
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return autoencoder
    
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    param_grid_autoencoder = {
        'units': [32,64,128,256,512],
        'units2': [0,8,16,32,64,128,256],
        'units3': [0,8,16,32,64,128,256],
        'dropout_rate': [0.2,0.3],
        "activation": ['relu'],
        "optimizer": ['adam'],
        'batch_size': [32,64],
        'epochs': [20,30,40,50]
        
}
    
    autoencoder_model = KerasClassifier(build_fn=create_autoencoder_model, verbose=1)
    grid_autoencoder = GridSearchCV(estimator=autoencoder_model, param_grid=param_grid_autoencoder,scoring='accuracy', cv=3)
    grid_result_autoencoder = grid_autoencoder.fit(X_train_resampled, y_train_resampled)
    
    
    results['Anto'] = {
        'best_params': grid_result_autoencoder.best_params_,
        'best_accuracy': grid_result_autoencoder.best_score_
    }
    
    
    print("Results:", results)
    
    best_model = grid_result_autoencoder.best_estimator_
    predictions = best_model.predict(X_test_scaled)
    binary_predictions = (predictions > 0.5).astype(int)
    
    
    accuracy = accuracy_score(y_test, binary_predictions)
    classification_rep = classification_report(y_test, binary_predictions)
    
    results['Anto']['test_accuracy'] = accuracy
    results['Anto']['classification_report'] = classification_rep
    
    message = "The Antoencoder process has completed successfully."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    
    formatted_anto_results = (
        f"The Antoencoder results:\n"
        f"Best Parameters: {results['Anto']['best_params']}\n"
        f"Best Accuracy: {results['Anto']['best_accuracy']:.4f}\n"
        f"Test Accuracy: {results['Anto']['test_accuracy']:.4f}\n"
        f"Classification Report:\n{results['Anto']['classification_report']}"
    )
    
    # Send the formatted message
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={formatted_anto_results}"
    print(requests.get(url).json())
    


except Exception as e:
    # Send a notification if an exception occurs
    message = "The Antoencoder process has stopped, there is an error."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
    error_message = f"An error occurred Autoencoder model: {str(e)}\n\n{traceback.format_exc()}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={error_message}"
    print(requests.get(url).json())

  
