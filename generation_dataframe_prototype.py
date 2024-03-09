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

# Définir la taille de l'échantillon (nombre d'années)
taille_echantillon = 1

# Définir les paramètres
prix_article_max = 100
prix_article_min = 10
prix_article_moyen = 75

prix_moyen_panier_clients = 125
prix_min_panier_clients = 50
prix_max_panier_clients = 200

quantite_moyen_panier_client = 4
quantite_min_panier_client = 1
quantite_max_panier_client = 10

# Progression de vente annuelle
croissance = 0.05  # 5% de croissance annuelle en moyenne
croissance = (1 + croissance) ** (1 / 52) - 1 #Passage sur la croissance sur la semaine

num_samples = 100


df_simulation = pd.DataFrame(columns=['Numero_Commande', 'Jour', 'Prix_Min_Article', 'Prix_Max_Article', 'Quantite_Article_Panier'])

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


# Enregistrement du DataFrame au format CSV
df_simulation.to_csv('prototype.csv', index=False)



Y = df_simulation[['Numero_Commande', 'Prix_Min_Article', 'Prix_Max_Article', 'Prix_Total_Panier', 'Quantite_Article_Panier']]

# Visualisation des données
plt.figure(figsize=(10, 6))
#plt.scatter(df_simulation['Jour'], Y['Prix_Min_Article'], label='Prix_Min_Article')
#plt.scatter(df_simulation['Jour'], Y['Prix_Max_Article'], label='Prix_Max_Article')
plt.scatter(df_simulation['Jour'], Y['Prix_Total_Panier'], label='Prix_Total_Panier')
#plt.scatter(df_simulation['Jour'], Y['Quantite_Article_Panier'], label='Quantite_Article_Panier')

plt.xlabel('Jour')
plt.ylabel('Valeurs')
plt.legend()
plt.show()



# Créer un histogramme de la valeur des paniers
plt.figure(figsize=(10, 6))
plt.hist(df_simulation['Prix_Total_Panier'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('prix')
plt.ylabel('Fréquence')
plt.title('Histogramme des prix total des paniers commandés')
plt.show()

# Générer la transaction
def generate_transaction():
    # Ajuster quantite_min_panier_client
    if quantite_min_panier_client < 4:
        quantite_min_adjusted = 4
    else:
        quantite_min_adjusted = quantite_min_panier_client - (quantite_min_panier_client % 4)
    
    quantite = random.randint(quantite_min_adjusted, quantite_max_panier_client + 2)
    price_min = random.uniform(prix_article_min, prix_article_max / 2)
    price_max = random.uniform(price_min, prix_article_max)
    total = random.uniform((price_min + price_max), prix_max_panier_clients + prix_min_panier_clients * 0.5)

    return {
        'Q': quantite,
        'Price_min_article': price_min,
        'Price_max_article': price_max,
        'Total': total,
        'Fraud': 0  # 0 pour les transactions non frauduleuses
    }

# Générer 100 transactions
transactions = [generate_transaction() for _ in range(100)]

# Créer la dataframe
df = pd.DataFrame(transactions)

for k in range(len(df)):
    if df['Q'].iloc[k] < int(quantite_min_panier_client* (1 - croissance)) or df['Q'].iloc[k] > int(quantite_max_panier_client * (1 + croissance)):
        df.at[k, 'Fraud'] = 1
    elif df['Total'].iloc[k] < prix_min_panier_clients * (1 + croissance) or df['Total'].iloc[k] > prix_max_panier_clients * (1 + croissance):
        df.at[k, 'Fraud'] = 1

df.to_csv('transaction_test/test.csv', index=False)
