import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.utils import parallel_backend

nltk.download(["punkt", "stopwords", "wordnet"])


def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X.values, Y.values, category_names


def tokenize(text):
    # normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize the text
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
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
        "clf__estimator__n_estimators": [10],
        "clf__estimator__min_samples_split": [2],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_macro', n_jobs=-1, verbose=12, cv=None)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.fit(X_test)
    for i, category_name in enumerate(category_names):
        report = classification_report(Y_test[:, i], Y_pred[:, i])
        print(f"The result for Category: {category_names}---->")
        print(report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        with parallel_backend('multiprocessing'):
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
