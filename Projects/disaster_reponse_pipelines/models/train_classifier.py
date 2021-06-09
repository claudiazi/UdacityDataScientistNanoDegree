import sys

import nltk
import numpy
from sqlalchemy import create_engine

nltk.download(["punkt", "wordnet"])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle

numpy.set_printoptions(threshold=sys.maxsize)


def load_data(database_filepath):
    """
    INPUT:
    database_filepath - the file path of the SQL database
    OUTPUT:
    X - a list of all messages text
    Y - 1 or 0 , depending on message category
    category_names - a list of  the message categories
    Loads the data from the SQLite database
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X.values, Y.values, category_names


def tokenize(text):
    """
    normalize, tokenize and lemmatize the text
    """
    # normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize the text
    words = word_tokenize(text)
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return words


def build_model():
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    parameters = {
        "clf__estimator__n_estimators": [10, 50, 100],
        "clf__estimator__min_samples_split": [2, 3],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, category_name in enumerate(category_names):
        report = classification_report(Y_test[:, i], Y_pred[:, i])
        print(f"The result for Category: {category_names}---->")
        print(report)


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
