# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:42:22 2021

@author: Manee44
"""
#Testing
"""
from flask import Flask

app=Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"
"""


import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category data for plotting
    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    
    # Plotting of Categories Distribution in Direct Genre
    direct_cate = df[df.genre == 'direct']
    direct_cate_counts = (direct_cate.mean()*direct_cate.shape[0]).sort_values(ascending=False)
    direct_cate_names = list(direct_cate_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Category Plotting
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # Categories Distribution
        {
            'data': [
                Bar(
                    x=direct_cate_names,
                    y=direct_cate_counts
                )
            ],

            'layout': {
                'title': 'Categories Distribution in Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in Direct Genre"
                }
            }
        }
    ]
    
    # Export Plotly to JASON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Load Plotly
    return render_template('Master.html', ids=ids, graphJSON=graphJSON)


#Displays Model Results
@app.route('/Go')
def go():
    # User Input
    query = request.args.get('query', '') 

    # Predict Classification
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Load Go.html 
    return render_template(
        'Go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()