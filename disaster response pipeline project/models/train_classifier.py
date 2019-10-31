#loading necessary packages
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
from sklearn.externals import joblib
import pdb

def load_data(database_filepath):
    '''
    input: database_filepath: the complete filepath where the database is stored
    
    output: returns messages from the message table,
    , the target variables and finally the category_names of the target variable
    
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql(con=engine,sql='select * from messages_tbl')
    df.set_index('id',inplace=True)
    X = df.loc[:,'message'].values
    Y = df.iloc[:,3:].values
    category_names = list(df.iloc[:,3:])                       
    return X, Y, category_names



def tokenize(text):
    '''
    input: this function takes the a bunch of text as an input(sentences, paragraphs etc..)
    
    output: returns lower case words after removing words that can be considered as stopwords
    
    '''
    tokens_ = word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    final_tokens = []
    for t in tokens_:
        t = str.lower(t)
        if t not in stopWords:
            lemma_ = WordNetLemmatizer()
            lemma = lemma_.lemmatize(word=t)
            final_tokens.append(t)
    return final_tokens
                          

def build_model():
    '''
    
    output: creates a pipeline of data transformations and applies a machine learning algorithm to it,
    with gridsearch and cv
   
    
    '''
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('multi', MultiOutputClassifier(AdaBoostClassifier()))
    ])
  
    parameters = {
     'tfidf__norm':['l1','l2'],
     'multi__estimator__learning_rate' :[0.75,1.0]
    }

    cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input: model: model object
           X_test: the test data without the target variable
           Y_test: the target variable
           category_names: category names of the target variable
    
    '''
    ypred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print('#####',col,'#####')
        print(classification_report(y_pred = ypred[:,i] ,y_true = Y_test[:,i]))
        print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], ypred[:,i])))
        print('\n')



def save_model(model, model_filepath):
    '''
    input: model object
    
    output: model pkl file
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #pdb.set_trace()
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        pdb.set_trace()
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