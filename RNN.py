# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:53:13 2023

@author: guery
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from tensorflow.keras.layers import Dense ,SimpleRNN, Dropout 
from keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv(r"C:\Users\guery\Desktop\Fraud detection\card_transaction_2.csv")

#######################################################################################
# Clean Data
#######################################################################################

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

#######################################################################################
# RNN
#######################################################################################

X, y = df.drop('Is Fraud?', axis=1), df['Is Fraud?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

def create_model(units=64,units2=64,units3=64, dropout_rate=0.2,activation='sigmoid',dropout_rate2=0.2,activation2='sigmoid',dropout_rate3=0.2,activation3='sigmoid',optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=(1, X_train_scaled.shape[1]), activation=activation, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(units2, activation= activation2,return_sequences=True))  
    model.add(Dropout(dropout_rate2))
    model.add(SimpleRNN(units3, activation= activation3))  
    model.add(Dropout(dropout_rate3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, verbose=0)

param_grid = {
    'units': [32, 64, 128],
    'units2': [32, 64, 128],
    'units3': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'dropout_rate2': [0.2, 0.3, 0.4],
    'dropout_rate3': [0.2, 0.3, 0.4],
    "activation": ['relu', 'leakyrelu'],
    "activation2": ['relu', 'leakyrelu'],
    "optimizer": ['adam', 'sgd']
}



grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

grid_result = grid.fit(X_train_reshaped, y_train)

print("Best Parameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)

best_RNN = grid_result.best_estimator_.model
rnn = best_RNN


