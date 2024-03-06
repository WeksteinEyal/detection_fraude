from flask import Flask, request, jsonify, send_file
import json
import numpy as np
from tensorflow import keras
import pickle
import ssl

app = Flask(__name__)

with open("jsonformatter.txt") as file:
    data = json.load(file)

timestamp = "5000_1709310489.792512"

# Load trained model
model = keras.models.load_model(f'models/{timestamp}/chat_model')

# Load tokenizer object
with open(f'models/{timestamp}/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open(f'models/{timestamp}//label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        inp = content['input']
        print(inp)

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
                print(response)
                return jsonify({'response': response})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('ssl/fraud-detector.ddns.net-chain.pem', 'ssl/new-fraud-detector.ddns.net-key.pem')
    
    app.run(debug=True, host='0.0.0.0', port=443, ssl_context=ssl_context)
    #app.run(debug=True, host='0.0.0.0', port=443, ssl_context=ssl_context)