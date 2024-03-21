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

if __name__ == '__main__':
    #ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    #ssl_context.load_cert_chain('ssl/fraud-detector.ddns.net-chain.pem', 'ssl/new-fraud-detector.ddns.net-key.pem')
    
    app.run(debug=True, host='0.0.0.0', port=5000)#port=443, ssl_context=ssl_context)
    #app.run(debug=True, host='0.0.0.0', port=443, ssl_context=ssl_context)