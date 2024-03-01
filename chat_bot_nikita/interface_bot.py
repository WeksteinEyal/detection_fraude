# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:46:53 2023

@author: guery
"""

import json 
import numpy as np
from tensorflow import keras

import colorama 
colorama.init()
from colorama import Fore, Style

import pickle

with open(r"jsonformatter.txt") as file:
    data = json.load(file)


def chat():
    timestamp = "5000_1709310489.792512"
    # load trained model
    model = keras.models.load_model(f'models/{timestamp}/chat_model')

    # load tokenizer object
    with open(f'models/{timestamp}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(f'models/{timestamp}//label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()