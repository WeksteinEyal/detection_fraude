# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:32:56 2024

@author: guery
"""

from flask import Flask, request, jsonify, send_file
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import pandas as pd
import time
from selenium.webdriver.chrome.options import Options

app = Flask(__name__)

# Set up Chrome options

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
    first_name = content['first_name']
    last_name = content['last_name']
    billing = content['billing']
    id = content['id']
    print(id)

    df_temp = df_reports.loc[(df_reports["first_name"]==first_name) & 
                             (df_reports["last_name"]==last_name) & 
                             (df_reports["billing"]==billing) & (df_reports["id"]==id)]
    if len(df_temp) > 0:
        return jsonify({'response': "You already reported this buyer."})
    else:
        df_reports.loc[len(df_reports)] = {"first_name": first_name, "last_name": last_name, "billing": billing, "id": id}
        df_reports.to_csv("list_reports.csv", sep=";", encoding="utf-8")
        return jsonify({'response': "Buyer reported."})

@app.route('/check', methods=['POST'])
def check():
    df_reports = pd.read_csv('list_reports.csv', sep=";", encoding="utf-8")
    content = request.get_json()
    first_name = content['first_name']
    last_name = content['last_name']
    billing = content['billing']

    df_temp = df_reports.loc[(df_reports["first_name"]==first_name) & 
                             (df_reports["last_name"]==last_name) & 
                             (df_reports["billing"]==billing)]
    response = f"This buyer has been reported {len(df_temp)} times."
    return jsonify({'response': response})
            

if __name__ == '__main__':
    #ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    #ssl_context.load_cert_chain('ssl/fraud-detector.ddns.net-chain.pem', 'ssl/new-fraud-detector.ddns.net-key.pem')
    
    app.run(debug=True, host='0.0.0.0', port=5000)#port=443, ssl_context=ssl_context)
    #app.run(debug=True, host='0.0.0.0', port=443, ssl_context=ssl_context)

