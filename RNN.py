# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:53:13 2023

@author: guery
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense ,SimpleRNN
from keras.models import Sequential
from sklearn.metrics import classification_report
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
# Définir une grille d'hyperparamètres à tester
units_values = [16, 32, 64, 128]
activation_values = ['relu', 'tanh', 'sigmoid', 'linear']

best_accuracy = 0
best_params = {}

# Itérer sur la grille d'hyperparamètres
for units, activation in product(units_values, activation_values):
    model = Sequential()
    model.add(SimpleRNN(units=units, activation=activation, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test), verbose=0)

    # Évaluer l'exactitude du modèle
    _, accuracy = model.evaluate(X_test_scaled, y_test)

    # Mettre à jour les meilleurs paramètres si l'exactitude actuelle est meilleure que la précédente
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {'units': units, 'activation': activation}
# Afficher les meilleurs paramètres et l'exactitude correspondante
print("Les meilleurs paramètres sont :", best_params)
print("La meilleure exactitude est :", best_accuracy)    

'''

best_params = {'units': 128, 'activation': 'relu'}
    
model = Sequential()
model.add(SimpleRNN(**best_params, input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0)      

loss,acc = model.evaluate(X_test_scaled, y_test)

print('Test accuracy:', acc)

y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary))




