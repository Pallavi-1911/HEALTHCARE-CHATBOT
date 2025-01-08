import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
import warnings
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 

warnings.filterwarnings("ignore", category=DeprecationWarning)


symptoms_dict = {} 
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def get_data():
    global symptoms_dict, severityDictionary, description_list, precautionDictionary

    try:
        with open('symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    severityDictionary[row[0]] = int(row[1])

        with open('symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    description_list[row[0]] = row[1]

        with open('symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 5:
                    precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

        print("Data loaded successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")

    df = pd.read_csv('Training.csv')
    global symptoms_dict
    symptoms_dict = {symptom: index for index, symptom in enumerate(df.columns[:-1])}

@app.route('/')
def chat():
    session.clear() 
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chatbot_response():
    global symptoms_dict
    user_message = request.json['message'].strip().lower()
    if 'name' not in session:
        session['name'] = user_message.capitalize()
        response = f"Hello, {session['name']}! Please enter the symptom you are experiencing."

    elif 'symptoms' not in session:
        symptoms_exp = re.findall(r'\b\w+\b', user_message)
        session['symptoms'] = symptoms_exp
        response = "Okay. From how many days are you experiencing these symptoms?"

    else:
        try:
            num_days = int(user_message)
            symptoms_exp = session['symptoms']
            predicted_disease = predict_disease(symptoms_exp, num_days)
            response = f"It might not be that bad but you should take precautions."
            response += f"\nYou may have {predicted_disease}."
            response += f"\n{description_list.get(predicted_disease, 'No description available')}"

            precautions = precautionDictionary.get(predicted_disease, [])
            response += "\nTake following measures:"
            for i, precaution in enumerate(precautions, 1):
                response += f"\n{i}. {precaution}"

        except ValueError:
            response = "Please enter a valid number for the duration of symptoms."

    return jsonify({'response': response})

def predict_disease(symptoms_exp, num_days):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_exp:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    return rf_clf.predict([input_vector])[0]


get_data()  
app.run()
