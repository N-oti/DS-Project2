# Disaster Response Pipeline Project


### Motivation

This project is to build a model for an API that classifies disaster messages.It will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Installation

This project written in pyathon and followig libraries need to be installed:

1. Sys
2. Panda
3. sqlalchemy
4. Nltk
5. Numpy
6. sklearn
7. json
8. plotly
9. Flask
10. Re
11. Pickle

### Project Components:

∙process_data.py: This code extracts the data from the input files messages.CSV ( which contains messages data) and categories.csv (which contains message categories) and load the cleaned merged data into SQLite database.

∙train_classifier.py: This code read the data from SQLite which is created by process_data.py and use the data to train the ML model to classify the message. The out put file from this process is a pickle file containing the fitted model.

∙templates folder: this folder contains all files required to run the web app.

∙disaster_messages and disaster_categories contain sample of messages and categries datasets in csv format.


### Run Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### License & Acknowledgements :

This app completed as part of Udacity Data Scientist Nanoodegree. teplates and data were provided by Udacity.