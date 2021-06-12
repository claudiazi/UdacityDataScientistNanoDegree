# Disaster Response Pipeline Project

### Motivation
This project is to build a simple web App using data from  [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. It can help organizations to quickly identify the most urgent and relevant messages.

### Application
An emergency worker input a new message and get classification results in several categories. 
![Screenshot of Web App - Classification](Classification.PNG)

The App will also display visualizations of the data.
![Screenshot of Web App - Visualisation](Visualisation.PNG)

## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- DisasterResponse.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
          |-- README
~~~~~~~


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
