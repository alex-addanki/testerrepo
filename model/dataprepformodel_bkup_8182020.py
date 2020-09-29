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
style.use('seaborn-poster')
style.use('ggplot')

from bs4 import BeautifulSoup
import requests
import os
import json

###from model.filepreprocess import preprocess
###from model.dataprepconfig import min_max_scaler,text_features, catagory_features,number_features,all_selected_features,eliminate_if_empty_list

from sklearn import preprocessing

global minval
global maxval
global min_max_scaler
global catagory_features
global number_features

min_max_scaler = preprocessing.MinMaxScaler()
text_features = ['keywords','original_title']
catagory_features = ['production_companies','Director_1','Director_2','Director_3','Actor_1','Actor_2','Actor_3','Actor_4','Actor_5','Genres_Action','Genres_Adventure','Genres_Animation','Genres_Comedy','Genres_Crime','Genres_Documentary','Genres_Drama','Genres_Family','Genres_Fantasy','Genres_Foreign','Genres_History','Genres_Horror','Genres_Music','Genres_Mystery','Genres_Romance','Genres_Thriller','Genres_War','Genres_Western']
number_features = ['budget', 'revenue','runtime','vote_average','vote_count','number_of_cast','number_of_director']
all_selected_features = ['budget', 'id', 'keywords','original_title', 'popularity','production_companies', 'revenue', 'runtime', 'vote_average','vote_count', 'Genres_Action', 'Genres_Adventure', 'Genres_Animation','Genres_Comedy', 'Genres_Crime', 'Genres_Documentary', 'Genres_Drama','Genres_Family', 'Genres_Fantasy', 'Genres_Foreign', 'Genres_History','Genres_Horror', 'Genres_Music', 'Genres_Mystery', 'Genres_Romance', 'Genres_Thriller','Genres_War', 'Genres_Western', 'number_of_cast', 'number_of_director','Director_1', 'Director_2', 'Director_3', 'Actor_1', 'Actor_2','Actor_3', 'Actor_4', 'Actor_5']
eliminate_if_empty_list = ['production_companies','Director_1','Director_2','Director_3','Actor_1','Actor_2','Actor_3','Actor_4','Actor_5','budget', 'revenue','runtime','vote_average','vote_count']



def preprocess(all5000_All):
    ##file = open('data\LX_categ_dataset_unfolded', 'rb')
    ##all5000_All = pickle.load(file)
    ##file.close()
    ##print('######################',type(all5000_All))
    all5000_All.drop(['cast', 'crew', 'Onlycast', 'Director'], axis=1)
    all5000_All = all5000_All.drop(["Genres_TV Movie"], axis=1)
    all5000_All = all5000_All.drop(["Genres_Science Fiction"], axis=1)

    return all5000_All

def dataprep(all5000_All):
    ##df= preprocess()
    df=all5000_All
    try:
        ##all5000_All.drop(['cast', 'crew', 'Onlycast', 'Director','keywords'], axis=1)
        all5000_All = all5000_All.drop(["Genres_TV Movie"], axis=1)
        all5000_All = all5000_All.drop(["Genres_Science Fiction"], axis=1)
        return all5000_All
    except:
        return df



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
    ###label_binarized_data = label_binarized_data.reshape(1,21120)
    return label_binarized_data


def preprocessing_text(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_text = tfidf_vectorizer.fit_transform(data)
    return tfidf_vectorized_text


# Plot features
# calculate mean
def meanbyfeature(data, feature_name, meanby_feature):
    mean_data = data.groupby(feature_name).mean()
    mean = mean_data[meanby_feature]
    mean_sort = mean.sort(meanby_feature, inplace=False, ascending=False)
    return mean_sort


def plot(data, kind, title, n_rows):
    plt.title(title, fontsize=15)
    data[:n_rows].plot(kind=kind)
    plt.show()


def show_features(database):
    print("\n",
          "--------------------------------------------------------------------------------------------------------")
    database.info()
    print("\n",
          "--------------------------------------------------------------------------------------------------------")

def preprocessing_catagory(data):
    data_c=0
    for i in range(len(catagory_features)):
        new_data = data[catagory_features[i]]
        print('CATEG DATA IS--->>>',new_data)
        new_data_c = preprocessing_categorical(new_data)
        if i == 0:
            data_c=new_data_c
        else:
            data_c = np.append(data_c, new_data_c, 1)
        print(i,data_c)
    return data_c

def preprocessing_numerical(data):
    data_list_numerical = list(zip(data['budget'], ###data['revenue'],
                                   data['runtime'], ###data['vote_average'],
                                   data['vote_count'], data['number_of_cast'],
                                   data['number_of_director']
                                   ))

    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical

def preprocessed_agregated_data(database):
    numerical_data   = preprocessing_numerical(database)
    print('numerical_data------->',numerical_data.shape)
    categorical_data = preprocessing_catagory(database)
    print('categorical_data------->',categorical_data.shape)
    all_data         = np.append(numerical_data, categorical_data, 1)
    return all_data


def preparefeatures(validation_df):
    print(validation_df)
    path = dataprep(validation_df)
    path['vote_average'] = path['vote_average'].astype(np.float64)
    path['vote_count']   = path['vote_count'].astype(np.float64)
    data = data_clean(path)

    target_gross = data['revenue'].to_frame()
    target_imdb_score = data['vote_average'].to_frame()

    database = data.drop('revenue', 1)
    database = data.drop('vote_average', 1)
    print('#################################################')
    print(database.columns)
    print(database.shape)
    print(len(database))
    print('#################################################')
    preprocessed_data = preprocessed_agregated_data(database)
    print('preprocessed_data.shape#######------->',preprocessed_data.shape)
    target_gross = preprocessing_numerical_minmax(target_gross)
    target_imdb_score = preprocessing_numerical_minmax(target_imdb_score)
    ###x = preprocessed_data.reshape((1,21125,-1))
    print(preprocessed_data)
    print("feature calculation complete\n")
    return preprocessed_data,target_gross,target_imdb_score

###x,y,z=preparefeatures()