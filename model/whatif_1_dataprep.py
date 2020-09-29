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
    file = open('data/DEMO_ALL_MOVIES_SMEAN', 'rb')
    all5000_all = pickle.load(file)
    file.close()
    return all5000_all

def wif1_load_VAG_files():
    file = open('data/DEMO_ALL_DIR', 'rb')
    dir = pickle.load(file)
    file.close()

    file = open('data/DEMO_ALL_ACTOR1', 'rb')
    act1 = pickle.load(file)
    file.close()

    file = open('data/DEMO_ALL_ACTOR2', 'rb')
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


def get_indx_of_movie(moviename):
  return datasource[datasource['original_title']==moviename].index.values[0]

def get_director_score(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Director_smean_enc"].iloc[locnv]

def get_actor1_score(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Actor1_smean_enc"].iloc[locnv]

def get_actor2_score(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Actor2_smean_enc"].iloc[locnv]

def get_actual_score(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["vote_average"].iloc[locnv]

def print_explainer(moviename):
  locnindx = get_indx_of_movie(moviename)
  exp = explainer.explain_instance(database2.values[locnindx], regr.predict, num_features = 8)
  exp.show_in_notebook(show_all=False) #only the features used in the explanation are displayed

def print_my_explainer(dir,a1,a2):
  locnindx = [dir,a1,a2]
  npa = np.asarray(locnindx, dtype=np.float32)
  exp = explainer.explain_instance(npa, regr.predict, num_features = 8)
  exp.show_in_notebook(show_all=False) #only the features used in the explanation are displayed



def get_director(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Director_1"].iloc[locnv]

def get_actor1(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Actor_1"].iloc[locnv]

def get_actor2(moviename):
  locnv = get_indx_of_movie(moviename)
  return datasource["Actor_2"].iloc[locnv]


############################################# Director_1	Director_smean_enc  Actor_1	Actor1_smean_enc

def get_director_score_byname(dirname):
  locnv = datasource_director[datasource_director['Director_1']==dirname].index.values[0]
  return datasource_director["Director_smean_enc"].iloc[locnv]

def get_actor1_score_byname(act1):
  locnv = datasource_act1[datasource_act1['Actor_1']==act1].index.values[0]
  return datasource_act1["Actor1_smean_enc"].iloc[locnv]

def get_actor1_score_byname(act2):
  locnv = datasource_act2[datasource_act2['Actor_2']==act2].index.values[0]
  return datasource_act2["Actor2_smean_enc"].iloc[locnv]




def wif1_preparefeatures(movietitle,director,actor1,actor2,budget):
    datasource          = wif1_load_database_file()
    datasource_director, datasource_act1, datasource_act2   =  wif1_load_VAG_files()

    locnv1 = datasource_director[datasource_director['Director_1'] == director].index.values[0]
    discore = datasource_director["Director_smean_enc"].iloc[locnv1]

    locnv2 = datasource_act1[datasource_act1['Actor_1'] == actor1].index.values[0]
    a1score = datasource_act1["Actor1_smean_enc"].iloc[locnv2]

    locnv3 = datasource_act2[datasource_act2['Actor_2'] == actor2].index.values[0]
    a2score = datasource_act2["Actor2_smean_enc"].iloc[locnv3]


    final_data_2_model = datasource[['Director_smean_enc', 'Actor1_smean_enc', 'Actor2_smean_enc']]

    print("feature calculation complete\n")

    new_array_2model = [[discore ,a1score,a2score]]
    explainer = lime.lime_tabular.LimeTabularExplainer(final_data_2_model.values,
                                                       mode='regression',
                                                       feature_names=final_data_2_model.columns,
                                                       discretize_continuous=True)


    ###print('preprocessed_data.shape#######------->', new_array_2model.shape)
    print('preprocessed_data.shape#######------->', new_array_2model)
    print("feature calculation complete\n")

    return new_array_2model,explainer,discore,a1score,a2score

