# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt


#df = pd.read_csv(r"C:\Users\guery\Desktop\Fraud detection\card_transaction.v1.csv")
'''
df.fillna("", inplace=True)
df['Amount'] = df['Amount'].str.replace('$', '')

fraudes = df[df['Is Fraud?'] == 'Yes' ]
non_fraudes = df[df['Is Fraud?'] == 'No']

non_fraudes = non_fraudes.sample(n = int(len(fraudes)))
df2 = pd.concat([fraudes, non_fraudes])
df2.to_csv('card_transaction_2.csv', index=False)
'''


df2 = pd.read_csv(r"C:\Users\guery\Desktop\Fraud detection\card_transaction_2.csv")
df2.info()
fraudes = df2[df2['Is Fraud?'] == 'Yes' ]
non_fraudes = df2[df2['Is Fraud?'] == 'No']

######################################################################################
# KNN architechture
######################################################################################

df2['Date'] = pd.to_datetime(df2[['Year', 'Month', 'Day']])
df2['Is Fraud?'] = df2['Is Fraud?'].replace({'Yes': 1, 'No': 0})

df2 = df2[['User','Date','Card','Amount','Merchant Name','Merchant State','Zip','MCC','Is Fraud?']]

X = df2[['User', 'Amount', 'MCC']]  # Sélection des caractéristiques pertinentes pour l'entraînement
y = df2['Is Fraud?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best = grid_search.best_params_['n_neighbors']

classifier = KNeighborsClassifier(n_neighbors=best)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")


############################################################################
# Fraude au cours du temps
############################################################################


# Créer une colonne 'Date' à partir des colonnes 'Day', 'Month' et 'Year'
df2['Date'] = pd.to_datetime(df2[['Year', 'Month', 'Day']])

# Créer un DataFrame pour les transactions frauduleuses
fraud_data = df2[df2['Is Fraud?'] == 'Yes']

# Grouper par date et compter le nombre de fraudes par jour
fraud_counts = fraud_data.groupby('Date').size()

# Tracer le graphique
plt.figure(figsize=(12, 6))
plt.plot(fraud_counts.index, fraud_counts.values, marker='o', linestyle='-')
plt.title('Nombre de fraudes au fil du temps')
plt.xlabel('Date')
plt.ylabel('Nombre de fraudes')
plt.grid(True)
plt.show()

######################################################################
#  Fraude par utilisateur
######################################################################

# Créer un DataFrame pour les transactions frauduleuses
fraud_data = df2[df2['Is Fraud?'] == 'Yes']

# Grouper par utilisateur et compter le nombre de fraudes par utilisateur
user_fraud_counts = fraud_data['User'].value_counts().reset_index()
user_fraud_counts.columns = ['User', 'Nombre de fraudes']

# Trier les utilisateurs par le nombre de fraudes décroissant
user_fraud_counts = user_fraud_counts.sort_values('Nombre de fraudes', ascending=False)

# Filtrer les utilisateurs avec une fraude au moins une fois
user_fraud_counts = user_fraud_counts[user_fraud_counts['Nombre de fraudes'] >= 1]

# Tracer le graphique à barres
plt.figure(figsize=(12, 6))
plt.bar(user_fraud_counts['User'], user_fraud_counts['Nombre de fraudes'])
plt.xlabel('Utilisateurs')
plt.ylabel('Nombre de fraudes')
plt.title('Nombre de fraudes par utilisateur')
plt.xticks(rotation=90)
plt.show()

###############################################################################
# Fraude par région
###############################################################################

# Créer un DataFrame pour les transactions frauduleuses
fraud_data = df2[df2['Is Fraud?'] == 'Yes']

# Grouper par utilisateur et compter le nombre de fraudes par utilisateur
user_fraud_counts = fraud_data['Merchant City'].value_counts().reset_index()
user_fraud_counts.columns = ['User', 'Nombre de fraudes']

# Trier les utilisateurs par le nombre de fraudes décroissant
user_fraud_counts = user_fraud_counts.sort_values('Nombre de fraudes', ascending=False)

# Filtrer les utilisateurs avec une fraude au moins une fois
user_fraud_counts = user_fraud_counts[user_fraud_counts['Nombre de fraudes'] >= 2]

# Tracer le graphique à barres
plt.figure(figsize=(12, 6))
plt.bar(user_fraud_counts['User'], user_fraud_counts['Nombre de fraudes'])
plt.xlabel('Ville')
plt.ylabel('Nombre de fraudes')
plt.title('Nombre de fraudes par Ville superieur à deux fois')
plt.xticks(rotation=90)
plt.show()

###############################################################################
# Nombre de fraude par MCC
###############################################################################

df2 = df2.dropna(subset=['MCC'])
# Créer un DataFrame pour les transactions frauduleuses
fraud_data = df2[df2['Is Fraud?'] == 'Yes']

# Grouper par utilisateur et compter le nombre de fraudes par utilisateur
user_fraud_counts = fraud_data['MCC'].value_counts().reset_index()
user_fraud_counts.columns = ['User', 'Nombre de fraudes']

# Trier les utilisateurs par le nombre de fraudes décroissant
user_fraud_counts = user_fraud_counts.sort_values('Nombre de fraudes', ascending=False)

# Filtrer les utilisateurs avec une fraude au moins une fois
user_fraud_counts = user_fraud_counts[user_fraud_counts['Nombre de fraudes'] >= 2]

# Tracer le graphique à barres
plt.figure(figsize=(12, 6))
plt.bar(user_fraud_counts['User'], user_fraud_counts['Nombre de fraudes'])
plt.xlabel('MCC')
plt.ylabel('Nombre de fraudes')
plt.title('Nombre de fraudes par MCC')
plt.xticks(rotation=90)
plt.show()

###############################################################################
# Nombre de fraude par ZIP
###############################################################################

df2 = df2.dropna(subset=['Zip'])
# Créer un DataFrame pour les transactions frauduleuses
fraud_data = df2[df2['Is Fraud?'] == 'Yes']

# Extraire les deux premiers chiffres du ZIP et ajouter trois zéros
fraud_data['Zip'] = fraud_data['Zip'].apply(lambda x: str(x)[:3] + '00')

# Grouper par code ZIP modifié et compter le nombre de fraudes par code ZIP
user_fraud_counts = fraud_data['Zip'].value_counts().reset_index()
user_fraud_counts.columns = ['Zip', 'Nombre de fraudes']

# Trier les codes ZIP par le nombre de fraudes décroissant
user_fraud_counts = user_fraud_counts.sort_values('Nombre de fraudes', ascending=False)

# Filtrer les codes ZIP avec au moins deux fraudes
user_fraud_counts = user_fraud_counts[user_fraud_counts['Nombre de fraudes'] >= 2]

# Tracer le graphique à barres
plt.figure(figsize=(12, 6))
plt.bar(user_fraud_counts['Zip'], user_fraud_counts['Nombre de fraudes'])
plt.xlabel('Code ZIP')
plt.ylabel('Nombre de fraudes')
plt.title('Nombre de fraudes par Code ZIP')
plt.xticks(rotation=90)
plt.show()

###############################################################################
# Nombre de fraude par montant
###############################################################################

df2 = pd.get_dummies(df2, columns=['Is Fraud?'])
df2['Is Fraud?_0'] = df2['Is Fraud?_0'].map({True: 1, False: 0})
df2['Is Fraud?_1'] = df2['Is Fraud?_1'].map({True: 1, False: 0})

df2 = df2.rename(columns={'Is Fraud?_0': 'Not Fraud', 'Is Fraud?_1': 'Fraud'})

info = df2.describe()

plt.figure(figsize=(10,6))
plt.scatter(fraudes.index, fraudes['User'], color='red', label='Fraudes')
#plt.scatter(non_fraudes.index, non_fraudes['User'], color='green', label='Non Fraudes')

plt.title('Montant des transactions($) par type ')
plt.xlabel('Index')
plt.ylabel('Montant')
plt.legend()


plt.show()

###############################################################################
# Algorithme de Luhn
###############################################################################

def luhn_validation(number):
    """
    Valide si un numéro de carte de crédit ou un numéro SIRET respecte l'algorithme de Luhn.
    """
    digits = [int(x) for x in str(number)]
    check_digit = digits.pop()
    digits.reverse()

    def luhn_checksum(digs):
        return (sum(digs[0::2]) + sum(sum(divmod(2 * d, 10)) for d in digs[1::2])) % 10

    return luhn_checksum(digits) == check_digit

# Exemple d'utilisation avec un numéro de carte de crédit
credit_card_number = "4532735188767663"
if luhn_validation(credit_card_number):
    print("Le numéro de carte de crédit est valide selon l'algorithme de Luhn.")
else:
    print("Le numéro de carte de crédit n'est pas valide selon l'algorithme de Luhn.")

# Exemple d'utilisation avec un numéro SIRET
siret_number = "35600000009075"
if luhn_validation(siret_number):
    print("Le numéro SIRET est valide selon l'algorithme de Luhn.")
else:
    print("Le numéro SIRET n'est pas valide selon l'algorithme de Luhn.")
    
    

