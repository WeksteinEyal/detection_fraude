# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:56:06 2024

@author: guery
"""

import pandas as pd

Name = 'Bobr'
Surname = 'Pinguin'
Email = 'bobr.pinguin@gmail.com'
Adress = 'Poland'
Id = 97986
button = 1

df = pd.read_csv("report.csv")
df2 = pd.read_csv('IdReport.csv')

def check_customer(Name, Surname, Email, Adress, df):
    if Name != '' and Surname != '' and Email != '' and Adress != '':
        for i in range(len(df)):
            if df.loc[i, "Name"] == Name and df.loc[i, "Surname"] == Surname and df.loc[i, "Email"] == Email and df.loc[i, "Adress"] == Adress:
                print(f'{Name} {Surname} with email: {Email} and address: {Adress} was {df.loc[i, "Number of report"]} time(s) reported.')
                print(f'You can report {Name} {Surname} with email: {Email} and address: {Adress}.')
                return  
        print("No matching records found.")
        print(f'You can report {Name} {Surname} with email: {Email} and address: {Adress}.')
    else:
        print("Please give Name, Surname, Email, and Address.")

check_customer(Name, Surname, Email, Adress, df)

def report(Name, Surname, Email, Adress, df, df2, button):
    if button == 1:
        if Name != '' and Surname != '' and Email != '' and Adress != '':
            for t in range(len(df2)):
                if Id == df2.loc[t, "Id"] and Name == df2.loc[t, "Name"] and Surname == df2.loc[t, "Surname"] and Email == df2.loc[t, "Email"] and Adress == df2.loc[t, "Adress"]:
                    print('You have already reported this information.')
                    return  
            
            df2.loc[len(df2)] = [Id, Name, Surname, Email, Adress]

            
            for k in range(len(df)):
                if df.loc[k, "Name"] == Name and df.loc[k, "Surname"] == Surname and df.loc[k, "Email"] == Email and df.loc[k, "Adress"] == Adress:
                    df.loc[k, "Number of report"] += 1
                    print(f'{Name} {Surname} with email: {Email} and address: {Adress} reported.')
                    return  
            
            df.loc[len(df)] = [Name, Surname, Email, Adress, 1]
            print(f'{Name} {Surname} with email: {Email} and address: {Adress} reported.')


report(Name, Surname, Email, Adress, df, df2, button)


df.to_csv("report.csv", index=False)
df2.to_csv("IdReport.csv", index=False)

