# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:32:56 2024

@author: guery
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService
import pandas as pd

driver = webdriver.Edge(service=EdgeService())

driver.get("https://whatismyipaddress.com/breach-check")

agree = driver.find_element(By.CLASS_NAME, "css-47sehv")
agree.click()

e = 'xxxxxxxxxxxxxxxxxx@gmail.com'

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

for c in df.columns:
    df[c] = df[c].apply(lambda x : x.split(':')[1])


