# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:30:15 2024

@author: guery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import random
from flask import Flask, request, jsonify, send_file
import json
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_file
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import pandas as pd
import time
from selenium.webdriver.chrome.options import Options
import random
from flask import Flask, request, jsonify, send_file
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents_0.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Freia"
print("Feel free to ask anything about fraud.")

# Définir la taille de l'échantillon (nombre d'années)
taille_echantillon = 1

def generate_decentered_beta_values(alpha, beta_, min_val, max_val, num_samples):
    # Générer des valeurs aléatoires selon une distribution bêta
    values = beta.rvs(alpha, beta_, size=num_samples)
    
    # Adapter les valeurs générées aux contraintes de min et max
    values = np.clip(values * (max_val - min_val) + min_val, min_val, max_val)
    
    return values

def get_param(min_value, max_value, mean_target, num_samples):
    # Marge d'erreur de 1%
    error_margin = mean_target * 0.01
    
    # Vérifier si la moyenne est proche du min ou du max et ajuster alpha ou beta en conséquence
    if mean_target < (min_value + max_value) / 2:    

        alpha_value = 0.01  
        beta_value = 5
        
        max_iterations = 100  # Nombre maximal d'itérations
        iterations = 0
        
        while iterations < max_iterations:
            random_values = generate_decentered_beta_values(alpha_value, beta_value, min_value, max_value, num_samples)
            current_mean = np.mean(random_values)
            if mean_target - error_margin <= current_mean <= mean_target + error_margin:
                break
            # Faire varier alpha
            alpha_value += 0.1  
            iterations += 1

        if iterations == max_iterations:
            beta_value -= 0.1
        
    elif mean_target == (min_value + max_value) / 2 :
        alpha_value = 5
        beta_value = 5
        
    else:

        alpha_value = 5
        beta_value = 0.01  
            
        max_iterations = 100  
        iterations = 0
        
        while iterations < max_iterations:
            random_values = generate_decentered_beta_values(alpha_value, beta_value, min_value, max_value, num_samples)
            current_mean = np.mean(random_values)
            if mean_target - error_margin <= current_mean <= mean_target + error_margin:
                break
            # Faire varier beta
            beta_value += 0.1  
            iterations += 1

        if iterations == max_iterations:
            alpha_value -= 0.1
        
    return alpha_value, beta_value

def val(min_value, max_value, mean_target, num_samples):
    alpha_value, beta_value = get_param(min_value, max_value, mean_target, num_samples)
    
    random_values = generate_decentered_beta_values(alpha_value, beta_value, min_value, max_value, num_samples)
    return random_values, alpha_value, beta_value

@app.route('/chat', methods=['POST'])
def chat():

    content = request.get_json()
    sentence = content['input']

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return jsonify({'response': response})

    else:
        return jsonify({'response': "I'm not sure to understand. Please try rephrasing."})

@app.route('/leaked', methods=['POST'])
def leaked():
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )

    driver.get("https://whatismyipaddress.com/breach-check")

    agree = driver.find_elements(By.TAG_NAME, "button")

    if len(agree) > 1:
        print('agreed')
        agree[2].click()

    print("clicked")



    content = request.get_json()
    e = content['input']

    email_input = driver.find_element(By.ID, "txtemail")
    email_input.send_keys(e)

    # Find the button and click on it
    btn = driver.find_element(By.ID, "btnSubmit")
    btn.click()

    df = pd.DataFrame({"Company Name":[],
                    "Domain Name":[],
                    "Date Breach":[],
                    "Type Info":[],
                    "Breach Overview":[],
                    "Total Number Affected":[]})

    blocks = driver.find_elements(By.CLASS_NAME, 'breach-wrapper')

    for block in blocks:
        domain = block.find_elements(By.CLASS_NAME, "breach-item")
        dictionnaire = {"Company Name" : domain[0].text,
                        "Domain Name" : domain[1].text,
                        "Date Breach" : domain[2].text,
                        "Type Info" : domain[3].text,
                        "Breach Overview" : domain[4].text,
                        "Total Number Affected" : domain[5].text}
        df.loc[len(df)] = dictionnaire
    driver.close()

    for c in df.columns:
        df[c] = df[c].apply(lambda x : x.split(':')[1])

    if len(df)== 0 :
        return jsonify({'response': "This email is not on the list of data breach."})
        
    else :
        response = ""
        for k in range(len(df)):
            response += 'Company Name : '+ df.loc[k, 'Company Name'] + "\n"
            response += 'Domain Name : '+ df.loc[k, 'Domain Name'] + "\n"
            response += 'Date Breach : '+ df.loc[k, 'Date Breach'] + "\n"
            response += 'Type Info : '+ df.loc[k, "Type Info"] + "\n"
            #response += 'Breach Overview :'+ df.loc[k, 'Breach Overview'] + "\n"
            #response += 'Total Number Affected : '+ df.loc[k, 'Total Number Affected'] + "\n"
            response += '###################################' + "\n"
            print(response)
        return jsonify({'response': response})



@app.route('/report', methods=['POST'])
def report():
    df_reports = pd.read_csv('list_reports.csv', sep=";", encoding="utf-8")
    content = request.get_json()
    first_name = content['first_name'].lower().replace(' ','')
    last_name = content['last_name'].lower().replace(' ','')
    billing = content['billing'].lower().replace(' ','')
    id = "A"+content['id']
    print(id)

    df_temp = df_reports.loc[(df_reports["first_name"]==first_name) & 
                             (df_reports["last_name"]==last_name) & 
                             (df_reports["billing"]==billing) & (df_reports["id_reports"]==id)]

    if len(df_temp) > 0:
        return jsonify({'response': "You already reported this buyer."})
    else:
        df_reports.loc[len(df_reports)] = {"first_name": first_name, "last_name": last_name, "billing": billing, "id_reports": id}
        df_reports = df_reports[['first_name','last_name','billing','id_reports']]
        df_reports.to_csv("list_reports.csv", sep=";", encoding="utf-8")
        return jsonify({'response': "Buyer reported."})

@app.route('/check', methods=['POST'])
def check():
    df_reports = pd.read_csv('list_reports.csv', sep=";", encoding="utf-8")
    content = request.get_json()
    first_name = content['first_name'].lower().replace(' ','')
    last_name = content['last_name'].lower().replace(' ','')
    billing = content['billing'].lower().replace(' ','')

    df_temp = df_reports.loc[(df_reports["first_name"]==first_name) & 
                             (df_reports["last_name"]==last_name) & 
                             (df_reports["billing"]==billing)]
    response = f"This buyer has been reported {len(df_temp)} times."
    return jsonify({'response': response})


    
@app.route('/parameters', methods=['POST'])
def parameters():
    try:
        content = request.get_json()

        # Définir les paramètres
        prix_article_max = float(content['prix_article_max'])
        prix_article_min = float(content['prix_article_min'])
        prix_article_moyen = float(content['prix_article_moyen'])

        prix_moyen_panier_clients = float(content['prix_moyen_panier_clients'])
        prix_min_panier_clients = float(content['prix_min_panier_clients'])
        prix_max_panier_clients = float(content['prix_max_panier_clients'])

        quantite_moyen_panier_client = float(content['quantite_moyen_panier_client'])
        quantite_min_panier_client = float(content['quantite_min_panier_client'])
        quantite_max_panier_client = float(content['quantite_max_panier_client'])

        # Progression de vente annuelle
        croissance = 0.01 * float(content['croissance'])  # 5% de croissance annuelle en moyenne
        print(prix_article_max,prix_article_min,prix_article_moyen,prix_moyen_panier_clients,prix_min_panier_clients,prix_max_panier_clients,quantite_moyen_panier_client,quantite_min_panier_client,quantite_max_panier_client,croissance)
        croissance = (1 + croissance) ** (1 / 52) - 1 #Passage sur la croissance sur la semaine

        num_samples = 100


        df_simulation = pd.DataFrame(columns=['Numero_Commande', 'Jour', 'Prix_Min_Article', 'Prix_Max_Article', 'Quantite_Article_Panier'])

        alpha1, beta1 = get_param(prix_article_min, prix_article_max, prix_article_moyen, num_samples)
        alpha_q, beta_q = get_param(quantite_min_panier_client, quantite_max_panier_client, quantite_moyen_panier_client, num_samples)

        jours = [] 
        prix_min_article = []
        prix_max_article = []
        quantite_article_panier = []
        numero_transaction = np.arange(1,num_samples*7 + 1)  
        # Simuler les transactions pour chaque jour de la semaine
        for jour in range(7):
            jours.append([jour] * num_samples)
                
            prix_article1 = generate_decentered_beta_values(alpha1, beta1, prix_article_min, prix_article_max, num_samples)
            prix_article2 = generate_decentered_beta_values(alpha1, beta1, prix_article_min, prix_article_max, num_samples)
            
            for i, j in zip(prix_article1, prix_article2):
                if i < j:
                    prix_min_article.append(i)
                    prix_max_article.append(j)
                else:
                    prix_min_article.append(j)
                    prix_max_article.append(i)            

            quantite_article_panier2 = generate_decentered_beta_values(alpha_q, beta_q,quantite_min_panier_client, quantite_max_panier_client, num_samples)
            
            # Arrondir chaque valeur à l'entier le plus proche
            quantite_article_panier2 = np.round(quantite_article_panier2).astype(int)
            
            quantite_article_panier.append(quantite_article_panier2)
                
        liste_J = []
        Q = []

        for sous_liste in jours:
            liste_J.extend(sous_liste)


        for sous_liste2 in quantite_article_panier :
            Q.extend(sous_liste2)

                        
        df_temp = pd.DataFrame({
            'Numero_Commande': numero_transaction,
            'Jour': liste_J,
            'Prix_Min_Article': prix_min_article,
            'Prix_Max_Article': prix_max_article,
            'Quantite_Article_Panier': Q,
        })

        df_simulation = pd.concat([df_simulation, df_temp], ignore_index=True)

        df_simulation['Quantite_Article_Panier'] = df_simulation['Quantite_Article_Panier'].astype(int)
        df_simulation['Prix_Total_Panier'] = 0


        # Boucle sur les transactions pour ajouter des valeurs à la colonne 'Prix_Total_Panier'
        for i in range(len(df_simulation)):
            if df_simulation.at[i, 'Quantite_Article_Panier'] == 1:
                df_simulation.at[i, 'Prix_Total_Panier'] = df_simulation.at[i, 'Prix_Min_Article']
                df_simulation.at[i, 'Prix_Max_Article'] = df_simulation.at[i, 'Prix_Min_Article']
                
            elif df_simulation.at[i, 'Quantite_Article_Panier'] == 2:
                df_simulation.at[i, 'Prix_Total_Panier'] = df_simulation.at[i, 'Prix_Min_Article'] + df_simulation.at[i, 'Prix_Max_Article']

            else:
                min_value = df_simulation.at[i, 'Prix_Min_Article'] + df_simulation.at[i, 'Prix_Max_Article'] + prix_article_min
                max_value = prix_max_panier_clients * (1 + croissance)
                mean_target = (min_value + max_value)/2
                
            
                random_values = val(min_value, max_value, mean_target, 1)
                
                df_simulation.at[i, 'Prix_Total_Panier'] = random_values[0]    

        info = df_simulation.describe()

        df_simulation = df_simulation[["Prix_Min_Article", "Prix_Max_Article", "Quantite_Article_Panier", "Prix_Total_Panier"]]

        # On ne normalise pas les données car la normalisation fausse la normalisation du panier client par la suite et la detection est faussée
        X_train, X_test, y_train, y_test = train_test_split(df_simulation, df_simulation, test_size=0.2, random_state=42)

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
        seuil_anomalie = np.mean(mse) / 1.5 #* np.std(mse)
        
        file_path = "to_send/seuil_user.txt"
        with open(file_path, "w") as file:
            file.write(str(seuil_anomalie))

        #Enregistrement du model
        import tensorflow as tf
        import tf2onnx
        import onnx
        input_signature = [tf.TensorSpec([None, 4], tf.float32, name='input_features')]
        onnx_model, _ = tf2onnx.convert.from_keras(autoencoder, input_signature, opset=13)
        onnx.save(onnx_model, "to_send/user_model.onnx")
        
        return send_file(
        "to_send/user_model.onnx",
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='user_model.onnx',
        )

    except Exception as e:
        print(str(e))

        return jsonify({'error': str(e)})
    
@app.route('/seuil', methods=['POST'])
def seuil():
    try:
        content = request.get_json()
        return send_file(
            "to_send/seuil_user.txt",
            mimetype='text/plain',
            as_attachment=True,
            download_name='seuil_user.txt'
        )

    except Exception as e:
        print(str(e))

        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
