import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management

# Suppress warnings for simplicity
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables
symptoms_dict = {}  # To store symptoms index mapping
severityDictionary = {}
description_list = {}
precautionDictionary = {}
clf = None  # Placeholder for classifier

# Function to load data from CSV files
def load_data():
    global symptoms_dict, severityDictionary, description_list, precautionDictionary, clf

    # Load symptom severity data
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])

    # Load symptom description data
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

    # Load symptom precaution data
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

    # Load symptoms from training data (for index mapping)
    df = pd.read_csv('Training.csv')
    global symptoms_dict
    symptoms_dict = {symptom: index for index, symptom in enumerate(df.columns[:-1])}

    # Train a Decision Tree classifier
    global clf
    clf = DecisionTreeClassifier()
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)

# Function to predict disease based on symptoms and duration
def predict_disease(symptoms_exp, num_days):
    global clf

    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_exp:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    return clf.predict([input_vector])[0]

# Route to render the chat interface
@app.route('/')
def chat():
    session.clear()  # Clear session to ensure a clean start
    return render_template('index.html')

# Route to handle chat interactions
@app.route('/chat', methods=['POST'])
def chatbot_response():
    global severityDictionary, description_list, precautionDictionary

    # Retrieve user input message
    user_message = request.json['message'].strip().lower()

    if 'name' not in session:
        # Assuming the first input is the user's name
        session['name'] = user_message.capitalize()
        response = f"Hello, {session['name']}! Please enter the symptom you are experiencing."

    elif 'symptoms' not in session:
        # Process symptoms input
        symptoms_exp = re.findall(r'\b\w+\b', user_message)
        session['symptoms'] = symptoms_exp
        response = "Okay. From how many days are you experiencing these symptoms?"

    else:
        # Process duration of symptoms
        try:
            num_days = int(user_message)
            symptoms_exp = session['symptoms']
            predicted_disease = predict_disease(symptoms_exp, num_days)

            # Prepare response
            response = "It might not be that bad but you should take precautions.\n"
            response += f"You may have {predicted_disease}.\n"
            response += f"{description_list.get(predicted_disease, 'No description available')}\n"

            precautions = precautionDictionary.get(predicted_disease, [])
            response += "Take following measures:\n"
            for i, precaution in enumerate(precautions, 1):
                response += f"{i}. {precaution}\n"

            session.clear()  # Clear session after diagnosis
        except ValueError:
            response = "Please enter a valid number for the duration of symptoms."

    return jsonify({'response': response})

if __name__ == '__main__':
    load_data()  # Load data on startup
    app.run(debug=True)
