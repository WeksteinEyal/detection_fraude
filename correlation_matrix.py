# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:53:13 2023

@author: guery
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


path_nikita = r"C:\Users\guery\Desktop\Fraud detection"
path_eyal = r"C:\Users\eyalw\Desktop\Cours\Projet d'étude\Database"
df = pd.read_csv(path_eyal + r"\card_transaction_2.csv")

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