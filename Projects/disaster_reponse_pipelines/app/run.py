import json

import numpy as np
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
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
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # count category
    categories = list(df.columns[4:])
    category_counts = []
    for category in categories:
        category_counts.append(np.sum(df[category]))

    def categories_distribution_in_genre(genre_name: str, df: pd.DataFrame) -> list:
        genre_df = df[df.genre == genre_name]
        genre_category_counts = []
        for category in categories:
            genre_category_counts.append(np.sum(genre_df[category]))
        return genre_category_counts

    social_category_counts = categories_distribution_in_genre("social", df)
    news_category_counts = categories_distribution_in_genre("news", df)
    direct_category_counts = categories_distribution_in_genre("direct", df)

    # create visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=categories, y=category_counts)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Categories"},
            },
        },
        {
            "data": [Bar(x=categories, y=direct_category_counts)],
            "layout": {
                "title": "Distribution of Message Categories in Direct Genre",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Categories"},
            },
        },
        {
            "data": [Bar(x=categories, y=news_category_counts)],
            "layout": {
                "title": "Distribution of Message Categories in News Genre",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Categories"},
            },
        },
        {
            "data": [Bar(x=categories, y=social_category_counts)],
            "layout": {
                "title": "Distribution of Message Categories in Social Genre",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Categories"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
