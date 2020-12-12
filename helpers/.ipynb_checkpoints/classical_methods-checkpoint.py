## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

## for explainer
# from lime import lime_text
## for word embedding
import gensim
import gensim.downloader as gensim_api


import sys
# sys.path.insert(1, './helpers')
from preprocessing import utils_preprocess_text
from summary import result_summary
import pickle


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import timeit

def webtable_classify(dtf, VECTORIZATION, CLASSIFIER):

    dtf['X'] = dtf['columns'] + ' ' +\
                dtf['values'] # + ' ' +\

    start_time = timeit.default_timer()


    lst_stopwords = nltk.corpus.stopwords.words("english")

    columns_to_clean = ['pg_title', 'section_title', 'table_caption', 
                        'columns', 'values']

    for column in columns_to_clean:
        dtf[column] = dtf[column].apply(lambda x: 
                  utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
                  lst_stopwords=lst_stopwords))
    # dtf.head()

    dtf["text_clean"] = dtf["X"].apply(lambda x: 
              utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
              lst_stopwords=lst_stopwords))

    ## split dataset
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
    ## get target
    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    if VECTORIZATION == 'TfidfVectorizer':
    ## Count (classic BoW)
        vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))
    else:
    ## Tf-Idf (advanced variant of BoW)
        vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))

    columns_to_clean = ['pg_title', 'section_title', 'table_caption', 
                        'columns', 'values']

    corpus = dtf_train["text_clean"]
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    # dic_vocabulary = vectorizer.vocabulary_

    from sklearn import feature_selection 
    y = dtf_train["y"]
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"], 
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()


    if VECTORIZATION == 'TfidfVectorizer':
        vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    else:
        vectorizer = feature_extraction.text.CountVectorizer(vocabulary=X_names)

    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    # dic_vocabulary = vectorizer.vocabulary_

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier

    if (CLASSIFIER == 'GradientBoostingClassifier'):
        classifier = GradientBoostingClassifier(random_state=0)
    elif CLASSIFIER == 'RandomForestClassifier':
        classifier = RandomForestClassifier(max_depth=10, random_state=0)
    else:
        classifier = naive_bayes.MultinomialNB()

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),  
                               ("classifier", classifier)])
    ## train classifier
    model["classifier"].fit(X_train, y_train)
    ## test
    X_test = dtf_test["text_clean"].values
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    elapsed = timeit.default_timer() - start_time
    #     print (elapsed)

    result_summary(y_test, predicted, predicted_prob, './fine_img/'+VECTORIZATION+'_'+CLASSIFIER+'_')


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        raise Exception('Input VECTORIZATION, CLASSIFIER, and dataframe path')

    VECTORIZATION = sys.argv[1]
    CLASSIFIER = sys.argv[2]
    dataframe_path = sys.argv[3]
    with open(dataframe_path, 'rb') as handle:
        dtf = pickle.load(handle)
    
    webtable_classify(, VECTORIZATION, CLASSIFIER)