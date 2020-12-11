#import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Load data function from message disaster table 
    
    Arguments:
    database_filepath -> Path to SQLite database where disaster messages is loaded
    
    Outpot:
    X -> dataframe containing messages
    Y -> dataframe containing features 
    categories -> columns names
    
    """
    engine = create_engine('sqlite:///' + database_filepath)
    #print("tables  :  " , engine.table_names())
    table_name='data/DisasterResponse'
    df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    categories=Y.columns
    
    return X,Y,categories


def tokenize(text):
    
    """
    tokenize the text
    
    Arguments:
       text -> text message which needs to be tekonized
       
    Output:
       clean_tokens -> list of cleaned tokens extracted from text
       
    """
    
    url_regex ='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #extract Urls from text
    detected_urls = re.findall(url_regex, text)
    
    #Replace URL with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Extract word token from text
    tokens = word_tokenize(text)
    
    #word Lemmanitizer
    lemmatizer = WordNetLemmatizer()

    #list of cleaned tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#Class to extract the starting verb of sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    StartingVerbExtractor this class to extract the starting verb of sentence
    as new feature for ML model
    
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Function to build model and create pipline
    
    Arguments:
    None
    
    Output:
    Return the model (cv)
    
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2,verbose=3 , n_jobs=-1 )
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate the model on test set and print the classification report
    
    Arguments:
    model -> ML pipline
    X_test -> Test features
    Y_test -> test labels
    category_names -> labels names (column names)
    
    Output:
    Print the classification report
    
    """
    
    y_pred=model.predict(X_test)
    for i , col in enumerate (Y_test):
        print(col)
        print(classification_report(Y_test[col],y_pred[:, i]))
    
    

def save_model(model, model_filepath):
    """
    Function to save the model
    
    Arguments:
    model - > the model
    model_filepath -> model file path
    
    Output:
    model file saved as pickle in the given path
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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