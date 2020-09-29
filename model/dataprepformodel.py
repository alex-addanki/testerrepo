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


def load_database_file():
    file = open('data\LX_categ_dataset_unfolded_database_array', 'rb')
    database = pickle.load(file)
    file.close()
    print('######################',type(database))
    print('######################', database.shape)
    return database

def load_preprocessed_file():
    file = open('data\LX_categ_dataset_unfolded_preprocessed_array', 'rb')
    preprocessed_data = pickle.load(file)
    file.close()
    print('######################',type(preprocessed_data))
    print('######################', len(preprocessed_data))
    print('######################', preprocessed_data.shape)
    return preprocessed_data

##def get_title_iloc(movietitle):
##    XX7 = database[database.original_title == movietitle].iloc[-1]
##    return XX7

def preparefeatures(movietitle):
    database          = load_database_file()
    preprocessed_data = load_preprocessed_file()
    movieiloc         = database[database.original_title == movietitle].iloc[-1]
    print('------->',movieiloc.name)
    preprocessed_data_predict = preprocessed_data[movieiloc.name]
    print('#######---> preprocessed_data_predict.shape=',preprocessed_data_predict.shape)
    return preprocessed_data_predict

