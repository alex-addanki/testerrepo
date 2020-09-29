import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import ast
import pickle

import dill
import lime
import numpy as np

import lime
import lime.lime_tabular
from sklearn.externals import joblib

style.use('seaborn-poster')
style.use('ggplot')

from bs4 import BeautifulSoup
import requests
import os
import json
import dill
###from model.filepreprocess import preprocess
###from model.dataprepconfig import min_max_scaler,text_features, catagory_features,number_features,all_selected_features,eliminate_if_empty_list

from sklearn import preprocessing

max_va     = 10.0  ###df["vote_average"].max()
max_gross  = 2550965087  ###df["gross"].max()
max_budget = 380000000  ###df["budget"].max()


global minval
global maxval
global min_max_scaler
global catagory_features
global number_features

min_max_scaler = preprocessing.MinMaxScaler()
text_features = []
catagory_features = []
number_features         = ['budget', 'runtime', 'Director_smean_enc', 'Actor1_smean_enc', 'Actor2_smean_enc']
all_selected_features   = number_features
eliminate_if_empty_list = number_features



def wif1_load_database_file():
    file = open('data/DEMO_ALL_MOVIED_RT', 'rb')
    all5000_all = pickle.load(file)
    file.close()
    return all5000_all

def wif1_load_VAG_files():
    file = open('data/DEMO_ALL_DIR_VAG', 'rb')
    dir = pickle.load(file)
    file.close()

    file = open('data/DEMO_ALL_ACT1_VAG', 'rb')
    act1 = pickle.load(file)
    file.close()

    file = open('data/DEMO_ALL_ACT2_VAG', 'rb')
    act2 = pickle.load(file)
    file.close()
    return dir,act1,act2

def data_clean(df):
    read_data = df
    select_data = read_data[all_selected_features]
    data = select_data.dropna(axis = 0, how = 'any', subset = eliminate_if_empty_list)
    data = data.reset_index(drop = True)
    for x in catagory_features:
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        data[y] = data[y].fillna(0.0).astype(np.float)
    return data


def preprocessing_numerical_minmax(data):
    global min_max_scaler
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data


def preprocessing_categorical(data):
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data)
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarized_data = label_binarizer.fit_transform(label_encoded_data)
    return label_binarized_data


def preprocessing_text(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_text = tfidf_vectorizer.fit_transform(data)
    return tfidf_vectorized_text

def preprocessing_catagory(data):
    data_c=0
    print('TYPE OF DATA COMMING IN = ',type(data))
    for i in range(len(catagory_features)):
        new_data = data[catagory_features[i]]
        new_data_c = preprocessing_categorical(new_data)
        if i == 0:
            data_c=new_data_c
        else:
            data_c = np.append(data_c, new_data_c, 1)
    print('TYPE OF DATA COMMING OUT = ',type(data_c))
    return data_c

def preprocessing_numerical(data):
    print('data.columns = ',data.columns)
    data_list_numerical = list(zip(data['budget'], data['runtime'],
                                   data['Director_smean_enc'],
                                   data['Actor1_smean_enc'],
                                   data['Actor2_smean_enc'],
                                   ))

    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical




def preprocessed_agregated_data(database):
    numerical_data   = preprocessing_numerical(database)
    print('numerical_data------->',numerical_data.shape)
    ##categorical_data = preprocessing_catagory(database)
    ###print('categorical_data------->',categorical_data.shape)
    all_data         = numerical_data ###np.append(numerical_data, categorical_data, 1)
    return all_data


##def get_title_iloc(movietitle):
##    XX7 = database[database.original_title == movietitle].iloc[-1]
##    return XX7

def wif1_preparefeatures(movietitle,director,actor1,actor2,budget):
    database          = wif1_load_database_file()
    dir, act1, act2   =  wif1_load_VAG_files()

    ###print('type(database)--->', type(database), str(len(database)))

    movieiloc7 = database[database.original_title == movietitle].iloc[-1]
    movieirowloc = int(movieiloc7.name)
    print('movieirowloc------->',movieirowloc)

    print('type(dir)--->',type(dir),str(len(dir)))

    ##preprocessed_data_predict = database.loc[database['original_title'] == movietitle]
    ###rebuild array
    preprocessed_data_predict = database
    print('preprocessed_data_predict------->', type(preprocessed_data_predict))
    if director != None :
        print('____________________________________')
        dir_val = dir._get_value(movieirowloc, 'Director_smean_enc')
        preprocessed_data_predict.at[movieirowloc,'Director_smean_enc'] = dir_val
        print('DIRECTOR#####################################',dir_val)
    if actor1 != None :
        print('____________________________________')
        act1_val = act1._get_value(movieirowloc, 'Actor1_smean_enc')
        preprocessed_data_predict.at[movieirowloc, 'Actor1_smean_enc'] = dir_val
        print('ACTOR-1#####################################',act1_val)
    if actor2 != None:
        print('____________________________________')
        act2_val = act2._get_value(movieirowloc, 'Actor2_smean_enc')
        preprocessed_data_predict.at[movieirowloc, 'Actor2_smean_enc'] = dir_val
        print('ACTOR-2#####################################',act2_val)

    if budget != None:
        print('____________________________________')
        ###budget = budget.replace('k','000')
        budget = int(budget)
        budget = budget/max_budget
        preprocessed_data_predict.at[movieirowloc, 'budget'] = budget
        print('BUDGET-2#####################################',budget)

    print('DONE HERE', len(preprocessed_data_predict))
    print('DONE HERE', preprocessed_data_predict.columns)

    final_data_2_model = preprocessed_data_predict[['budget', 'runtime', 'Director_smean_enc', 'Actor1_smean_enc',  'Actor2_smean_enc']]

    print('DONE HERE',len(final_data_2_model))
    print('DONE HERE-------', final_data_2_model)
    print('DONE HERE', final_data_2_model.columns)


    path = final_data_2_model
    data = data = data_clean(path)

    print('cleaning done')

    preprocessed_data = preprocessed_agregated_data(final_data_2_model)

    print('preprocessed_data.shape#######------->', preprocessed_data.shape)
    ###print('preprocessed_data.shape#######------->', preprocessed_data)
    print("feature calculation complete\n")

    new_array_2model = preprocessed_data[movieirowloc]

    ####x7 = dill.load(open('model/LIME_EXP_RandomForrestRegressorModel_VAVG_21082020_213610.exp', 'rb'))
    ####print('#######------->pkl file loaded', x7)
    explainer = lime.lime_tabular.LimeTabularExplainer(final_data_2_model.values,
                                                       mode='regression',
                                                       feature_names=final_data_2_model.columns,
                                                       ###categorical_features = [3],
                                                       ##categorical_names = ['CHAS'],
                                                       discretize_continuous=True)



    print('preprocessed_data.shape#######------->', new_array_2model.shape)
    print('preprocessed_data.shape#######------->', new_array_2model)
    print("feature calculation complete\n")

    return new_array_2model,explainer

