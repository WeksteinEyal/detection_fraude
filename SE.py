# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:32:56 2024

@author: guery
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth

import pandas as pd
import time
#driver = webdriver.Edge(service=EdgeService().install()))

start_time = time.time()
from selenium.webdriver.chrome.options import Options

# Set up Chrome options
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
driver.get_screenshot_as_file("screen.png")

agree = driver.find_elements(By.TAG_NAME, "button")
driver.get_screenshot_as_file("screen2.png")

if len(agree) > 1:
    agree[2].click()

e = 'eyalrw@yahoo.fr'

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

if len(df)== 0 :
    print("This email is not on the list of data breach.")
    
else :
    for k in range(len(df)):
        print('Company Name : ', df.loc[k, 'Company Name'])
        print('Domain Name : ', df.loc[k, 'Domain Name'])
        print('Date Breach : ', df.loc[k, 'Date Breach'])
        print('Type Info : ', df.loc[k, "Type Info"])
        print('Breach Overview :', df.loc[k, 'Breach Overview'])
        print('Total Number Affected : ', df.loc[k, 'Total Number Affected'])
        print('###################################')
        
driver.close()

end_time = time.time()
duration = end_time - start_time
print(f"Execution time: {duration:.2f} seconds")