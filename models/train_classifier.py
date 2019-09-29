#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
import re
import sys
import pickle

import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def build_pipeline(**kwargs):
    """
    Builds machine learning pipeline, sets parameters for components if given.
    :param kwargs: key value arguments pipeline parameters
    :return sklearn nlp pipeline
    """

    tf_args = {}
    clf_args = {}
    vect_args = {}

    classifier = kwargs.get('classifier', DecisionTreeClassifier())

    for key, value in kwargs.items():
        if key.startswith('tf'):
            tf_args.update({key.split('__')[-1]: value})
        elif key.startswith('clf'):
            clf_args.update({key.split('__')[-1]: value})
        elif key.startswith('vect'):
            vect_args.update({key.split('__')[-1]: value})

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, **vect_args)),
        ('tfidf', TfidfTransformer(**tf_args)),
        ('clf', MultiOutputClassifier(classifier.set_params(**clf_args)))
    ])

    return pipeline


def load_model(filename):
    """
    Returns model dumped in pickle file
    :param filename: model pickle file
    :return model in pickle
    """
    return pickle.load(open(filename, 'rb'))


# load data from database
def load_data(dbname, tablename='messages'):
    """
    Loads data saved in db.table
    :param dbname: sql db name
    :param tablename: sql table name
    :return input and label data as tuple
    """
    engine = create_engine('sqlite:///{}'.format(dbname))
    df = pd.read_sql_table(tablename, con=engine)
    X = df['message']
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)
    return X, Y


def tokenize(text):
    """
    Removes special characters, stopwords
    First lemmatizes and then stems the tokens
    :param text: free text
    :return tokenized words
    """

    text = re.sub('[^(a-zA-Z0-9)]', ' ', text.lower())
    words = word_tokenize(text)
    clean = [word for word in words if word not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in clean]
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


def get_best_estimator(pipeline, param_grid, X_train, Y_train):
    """
    Implements grid search for given param_grid on pipeline
    :param pipeline: sklearn pipeline
    :param param_grid: Dictionary with parameters names and values
    :param X_train: Training Input Data
    :param Y_train: Training Label Data
    :return best estimator
    """

    cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
    cv.fit(X_train, Y_train)

    print("Grid Search results for pipeline:\n best parameters : {}".
          format(cv.best_params_))
    return cv.best_estimator_


def report_scores(pipeline, X_test, Y_test):
    """
    Runs the model on test dataset
    Calculates f1 score, precision and recall for each category
    :param pipeline: trained ML pipeline
    :param X_test: Test Input Data
    :param Y_test: Test Label Data
    :returns pandas dataframe with metrics for each class of each category
    """

    Y_preds = pipeline.predict(X_test)
    Y_preds = pd.DataFrame(Y_preds, columns=Y_test.columns)

    report = []
    for col in Y_test.columns.tolist():
        report.append({
            'category': col,
            'precision':
                precision_score(Y_test[col], Y_preds[col], average='micro'),
            'recall':
                recall_score(Y_test[col], Y_preds[col], average='micro'),
            'f1_score':
                f1_score(Y_test[col], Y_preds[col], average='micro')})

    df_report = pd.DataFrame(report)
    return df_report


def save_model(model, filename):
    """
    Saves given model as pickle file
    :param model: ML model
    :param filename: output pickle file path
    """
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.
              format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_pipeline(
            classifier=RandomForestClassifier(random_state=13,
                                              n_estimators=50))

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        print(report_scores(model, X_test, Y_test))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
