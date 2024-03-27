# Fraud Block
🇬🇧 [English version](README.md)
## Introduction
Le projet Fraud Block est un outil de sécurité open source pour les e-commerçants. Son objectif est d'aider à identifier les transactions anormales pour une plateforme de commerce électronique. Il comble le vide des outils gratuits qui peuvent être mis à la disposition de tous pour se protéger contre les transactions frauduleuses, en utilisant uniquement des informations non sensibles.

L'application est compatible avec Android, iOS, Windows et MacOS. En raison de contraintes de financement, elle n'est actuellement disponible pour un accès public que sur les plates-formes [Android](#android) et [Windows](#windows).

Ce projet a été initié dans le cadre des projets de master à ESME Paris, une école d'ingénieurs française. Les auteurs sont *Illan Assuied*, *[Nikita Guery](https://github.com/gostravel)*, *Paul-Henri Lamarque*, et *[Eyal Wekstein](https://github.com/WeksteinEyal)*.

## Index
- [Premiers pas](#getting-started) : Pour comprendre rapidement comment utiliser l'outil.
- [Documentation d'utilisation](#usage-documentation) : Pour obtenir tous les détails.
- [Liens de téléchargement](#download) : Pour télécharger l'application.

## Premiers pas
On peut utiliser le [détecteur de fraude](#fraud-detector) après avoir rempli et envoyé ses informations e-commerce depuis l'onglet [Profil](#profile). Ces informations sont utilisées pour construire un modèle d'intelligence artificielle adapté aux besoins e-commerce de l'utilisateur, capable de détecter les transactions anormales. Les données ne sont pas stockées et le modèle est enregistré sur l'appareil de l'utilisateur. Cette étape nécessite une connexion Internet. Ensuite, l'utilisateur peut utiliser le détecteur de fraude sans Internet en saisissant les informations d'une transaction suspecte.

Un [chatbot](#chatbot) est disponible pour accompagner l'utilisateur pendant l'utilisation de l'application, ainsi que pour lui fournir des informations sur le sujet des fraudes.

Depuis l'onglet [Services](#services), l'utilisateur peut également vérifier si un e-mail est répertorié sur un ou plusieurs sites affectés par une fuite de données. Il est également possible de signaler un utilisateur ou de vérifier combien de fois un utilisateur a été signalé.

## Documentation d'utilisation
### Profil
<img src="assets/profile_top.jpeg" alt="Partie supérieure de l'onglet Profil" width="150" height="auto"> <img src="assets/profile-bottom.jpeg" alt="Partie inférieure de l'onglet Profil" width="150" height="auto">


Le point d'entrée du détecteur de fraude. Ici, l'utilisateur saisit ses informations e-commerce pour créer ultérieurement un modèle de détection de fraude adapté à son marché. Cette étape est nécessaire pour utiliser le détecteur de fraude et nécessite une connexion Internet.

Voici une liste exhaustive de tous les champs requis. Toutes les valeurs sont numériques et peuvent être des entiers ou des décimales.
|Champ|Description|
|---|---|
|Average item price|Le prix moyen de tous les différents articles en vente|
|Minimum item price|L'article le moins cher en vente|
|Maximum item price|L'article le plus cher en vente|
|Average basket price|Le prix moyen du panier de tous les différents articles vendus|
|Minimum basket price|Le panier le moins cher vendu|
|Maximum basket price|Le panier le plus cher vendu|
|Average basket quantity|La quantité moyenne d'articles par panier parmi tous les différents articles vendus|
|Minimum basket quantity|La plus petite quantité d'articles vendus|
|Maximum basket quantity|La plus grande quantité d'articles vendus|
|Growth rate %|Le taux de croissance e-commerce en % (annuel)|

### Détecteur de fraude
<img src="assets/detector.jpeg" alt="Onglet Détecteur de fraude" width="150" height="auto">

Après avoir soumis son profil de marché, l'utilisateur peut remplir les champs suivants pour vérifier si une transaction est potentiellement frauduleuse ou non. Ne nécessite pas de connexion Internet.
|Champ|Description|
|---|---|
|Minimum item price|Le moins cher des articles dans le panier du client|
|Maximum item price|Le plus cher des articles dans le panier du client|
|Quantity in basket|Le nombre d'articles dans le panier du client|
|Total basket price|Le montant total payé par le client pour ce panier|

### ChatBot
<img src="assets/chatbot.jpeg" alt="Onglet ChatBot" width="150" height="auto">

Le ChatBot a été formé pour répondre aux questions sur l'application et sur le sujet des fraudes. Il ne fonctionne qu'en anglais et n'est pas un chatbot génératif (comme ChatGPT). Cela signifie que vous pouvez soumettre une question sans vous soucier de la fiabilité de la réponse. Toutes les réponses ont été rédigées et approuvées par l'équipe. Le ChatBot nécessite une connexion Internet.

### Services
<img src="assets/services.jpeg" alt="Onglet Services" width="150" height="auto">

Les services sont des outils supplémentaires pour vérifier si un client est suspect. Il est conseillé de les utiliser si une fraude potentielle a été détectée.
#### Vérificateur d'e-mails
Cet outil est utilisé pour vérifier si un e-mail est répertorié sur un site ayant connu une fuite de données. L'utilisateur peut ensuite consulter tous les sites pertinents, la date de la fuite de données, le nombre de données divulguées et une description de l'incident. Il nécessite une connexion Internet et est alimenté par [whatismyipaddress](https://whatismyipaddress.com/).

#### Système de signalement
Ce système de signalement collaboratif a pour but de signaler et de vérifier les clients suspects, en utilisant les informations de l'adresse de facturation (nom, prénom, adresse).

## Téléchargement
Malheureusement, malgré sa compatibilité avec iOS et MacOS, et comme nous ne l'avons pas téléchargée sur un magasin d'applications, l'application ne peut actuellement pas être utilisée sur iOS et MacOS.
### Android
[lien de téléchargement](https://drive.google.com/uc?export=download&id=1Rf_cqEEIspfsSDNWcY4MLHr016_C7RwI)
### Windows
[lien de téléchargement](https://drive.google.com/uc?export=download&id=1joEPXb3MOEKN5G9aDT22bvMbUvYnoYfL)

Pour installer Fraud Block sur votre machine Windows :

1) Faites un clic droit sur le fichier zip téléchargé **Fraud_Block_1.0.x.x.zip**
<img src="assets/install_fraudblock_0.png" alt="installation 1" width="400" height="auto">

2) Cliquez sur "Extraire tout..."
<img src="assets/install_fraudblock_-1.png" alt="installation 2" width="400" height="auto">

3) Vous pouvez choisir un dossier de destination pour le répertoire d'installation ou le laisser tel quel. Cliquez sur "Extraire".
<img src="assets/install_fraudblock_-2.png" alt="installation 3" width="400" height="auto">

4) Ouvrez le dossier **Fraud_Block_1.0.x.x**
<img src="assets/install_fraudblock_-3.png" alt="installation 4" width="400" height="auto">

5) Ouvrez le certificat de sécurité **Fraud_Block_1.0.x.x_x64.cer**.
<img src="assets/install_fraudblock_1.png" alt="installation 5" width="400" height="auto">

6) Cliquez sur "Installer un certificat...".
<img src="assets/install_fraudblock_2.png" alt="installation 6" width="400" height="auto">

7) Sélectionnez "Ordinateur local" puis cliquez sur "Suivant".
<img src="assets/install_fraudblock_3.png" alt="installation 7" width="400" height="auto">

8) Sélectionnez "Placer tous les certificats dans le magasin suivant" puis cliquez sur "Parcourir...".
<img src="assets/install_fraudblock_4.png" alt="installation 8" width="400" height="auto">

9) Sélectionnez "Personnes autorisées" puis cliquez sur "OK".
<img src="assets/install_fraudblock_5.png" alt="installation 9" width="400" height="auto">

10) Cliquez sur "Suivant".
<img src="assets/install_fraudblock_6.png" alt="installation 10" width="400" height="auto">

11) Cliquez sur "Terminer".
<img src="assets/install_fraudblock_7.png" alt="installation 11" width="400" height="auto">

12) Cliquez sur "OK" puis sur "OK" à nouveau.
<img src="assets/install_fraudblock_8.png" alt="installation 12" width="400" height="auto">

13) Ouvrez le fichier **Fraud_Block_1.0.x.x_x64.msix**.
<img src="assets/install_fraudblock_9.png" alt="installation 13" width="400" height="auto">

14) Cliquez sur "Installer".
<img src="assets/install_fraudblock_10.png" alt="installation 14" width="400" height="auto">

15) Vous pouvez maintenant accéder à l'application en tapant "Fraud Block" dans la recherche Windows.
<img src="assets/install_fraudblock_11.png" alt="installation 15" width="400" height="auto">

16) Remplissez votre profil et testez toutes les transactions qui vous semblent anormales.
<img src="assets/install_fraudblock_12.png" alt="installation 16" width="400" height="auto">
