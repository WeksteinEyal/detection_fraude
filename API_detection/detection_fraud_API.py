# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:22:55 2024

@author: guery
"""
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify, send_file
import json
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

################################################################################################
#
#   Data E-Commerçant
#
################################################################################################

app = Flask(__name__)

@app.route('/detection', methods=['POST'])
def parameters():

    prix_article_max = 100
    prix_article_min = 10
    prix_article_moyen = 50

    prix_moyen_panier_clients = 100
    prix_min_panier_clients = 20
    prix_max_panier_clients = 200

    quantite_moyen_panier_client = 3
    quantite_min_panier_client = 1
    quantite_max_panier_client = 10

    # Progression de vente annuelle
    croissance = 0.05  # 5% de croissance annuelle en moyenne

    # Nombre de transactions journalières
    nombre_transactions_journalieres_min = 5
    nombre_transactions_journalieres_max = 20
    nombre_transactions_journalieres_moyen = 12

    ################################################################################################
    #
    #   Data Panier Client
    #
    ################################################################################################

    prix_article_plus_cher = 10
    prix_article_moin_cher = 10
    quantite_article_panier = 10
    prix_panier = 100

    ################################################################################################
    #
    #   Algorithme de prediction Autoencoder
    #
    ################################################################################################

    df = pd.read_csv("prototype.csv")
    df = df[["Prix_Min_Article", "Prix_Max_Article", "Quantite_Article_Panier", "Prix_Total_Panier"]]

    # On ne normalise pas les données car la normalisation fausse la normalisation du panier client par la suite et la detection est faussée
    X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.2, random_state=42)

    autoencoder = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(4, activation='sigmoid')
    ])

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

    predictions = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)

    # Détermination du seuil d'anomalie (par exemple, en utilisant la moyenne + 2*écart-type des erreurs)
    seuil_anomalie = np.mean(mse) + 2 * np.std(mse)

    panier_client = np.array([prix_article_moin_cher, prix_article_plus_cher, prix_panier, quantite_article_panier])
    panier_client_normalized = (panier_client.reshape(1, -1))

    reconstruction = autoencoder.predict(panier_client_normalized)

    mse_panier_client = np.mean(np.square(panier_client_normalized - reconstruction))

    if mse_panier_client > seuil_anomalie:
        print("Potential Fraud detected!")
        if quantite_article_panier > quantite_max_panier_client :
            print("The quantity of article in Basket is anormaly more than usual,be carefull")
        elif quantite_article_panier < quantite_min_panier_client :
            print("The quantity of article in Basket is anormaly less than usual,be carefull")
            
        elif prix_panier > prix_max_panier_clients :
            print("The Basket Total Price is anormaly more than usual,be carefull")
            
        elif prix_panier < prix_min_panier_clients :
            print("The Basket Total Price is anormaly less than usual,be carefull")
            
        else :
            print("A product is purchased more or less than usual,be carefull")
            
    else:
        print("Normal Transaction")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
