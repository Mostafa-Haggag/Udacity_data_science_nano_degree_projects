import sys
import nltk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os
import re
from scipy.stats import gmean
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,fbeta_score, make_scorer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def load_data(database_filepath):
"""
Description of the function:
load the data cleaned data
Parameters:
database_filepath this the path of the cleaned data
Returns:
X message
Y 
category_names columns
"""
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
"""
Description of the function:
this takes the texts message and tokanize it
Parameters:
text the message
Returns:
for reach message i return the toknized messages
"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
"""
Description of the function:
The pipeline
Parameters:
No input
Returns:
The optimized parameters for this pipeline
"""
    pipeline0 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}
    cv = GridSearchCV(pipeline0, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
"""
Description of the function:
This evalute the model
Parameters:
model the trianed optimized model
x_test x test category
y_test the labels
category_names
Returns:
nothing
"""
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
"""
Description of the function:

Parameters:.
model the model
model_filepath the path we want to save the model
Returns:

"""
   with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()