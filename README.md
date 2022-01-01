# DAND_Project2
This Project is part of Udacity course submitions. Udacity DataScience Nanodegree Project 2 - Disaster Response System.


# Project Motivation
In this project we apllied most important pipelines:

## 1-ETL Pipeline:
- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

## 2-ML Pipeline:
-Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds machine learning pipeline
- Trains model using GridSearchCV
- Exports the final model as a pickle file

# Libraries
The libraries that used in this project:

- pandas
- re
- sys
- Flask
- sqlite3
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- plotly

# Files Structure
The files structure:

-App
- Run.py: flask file to run the app
  - \Templates
  - Master.html: main page of the web application 
  - Go.html: result web page

-Data
  - categories.csv: categories dataset
  - messages.csv: messages dataset
  - DisasterResponse.db: disaster response database
  - process_data.py: ETL process

-Models
  - train_classifier.py: classification code

-NoteBooks
  - ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
  - ML Pipeline Preparation.ipynb: contains ML pipeline preparation code

- README.md: read me file


# Run The App Instruction
- 1-Make sure that you added port number 3001 in your firewall.
- 2-Run following comand to process data ETL,{`python Data/process_data.py Data/messages.csv Data/categories.csv Data/DisasterResponse.db`} .
- 3-Run train_classifier.py,{ `python Models/train_classifier.py Data/DisasterResponse.db Models/classifier.pkl`} .
- 4-Open your CMD on App directory.
- 5-Set Flask_App varible to Run.py,{Set Flask_App=Run.py}.
- 6-Use flask run command,{flask run}.
- 7-Go to http://0.0.0.0:3001/ using web browser.
