from dash.dependencies import Input, Output , State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from joblib import load
import numpy as np
import pandas as pd

from app import app

from model import filepreprocess

from model import dataprepformodel
from model.dataprepformodel import preparefeatures
import joblib

from model.prediction_explainability import load_prediction_explainability_metrics



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
import pickle
import time

filename='model\DecisionTreeRegressorModel_IMDB18082020_022901.h5'
###filename='model/whatif_models/RandomForrestRegressorModel_VAVG_21082020_213610.h5'

###filename='model\RandomForrestRegressorModel_IMDB18082020_022431.h5'
##df = pd.read_csv('data\dataset_model.csv')
df= filepreprocess.preprocess()

movie_titles=df['original_title'].unique().tolist()

##movie_titles = np.sort(movie_titles).tolist()

###style = {'padding': '1.5em'}


layout = dcc.Loading(
    html.Div([
    dcc.Markdown('###### Select the Movie'),
    dcc.Dropdown(
        id='movie-dropdown',
        options=[{'label': k, 'value': k} for k in movie_titles],
        value='Movie-Title'
    ),

    html.Hr(),

    html.Button(id='predict-button-state', n_clicks=0, children='Predict-IMDB-Score'),

    html.Div(id='output-state'),
    html.Div(id='output-state-2')

]) , type='cube', fullscreen=True)


@app.callback(Output('output-state', "children"),
              [Input('predict-button-state', 'n_clicks')],
              [State('movie-dropdown', 'value')])
def predict_output(n_clicks, input1):
    ###time.sleep(3)
    if input1 == "Movie-Title":
        return u''
    else:
        validation_df = input1 ###df[df['original_title'] == input1]
        ##print('#######------->',type(validation_df))
        ##print('#######------->', validation_df.columns)
        ###preprocessed_data,target_gross,target_imdb_score = preparefeatures(validation_df)
        preprocessed_data = preparefeatures(validation_df)
        ###print(preprocessed_data.shape)
        print('#######-------> Features Generated')
        loaded_model = pickle.load(open(filename,"rb"))
        print('#######------->pkl file loaded')
        result = loaded_model.predict(preprocessed_data.reshape(1, -1))
        importance = loaded_model.feature_importances_
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        print(result[0])
        print(type(result))
        imdb_score=round(result[0] * 10,1)
        print('imdb_score=',result)
        metrics_return=load_prediction_explainability_metrics(validation_df,result)
        print('metrics_return Completed .......')

        returnmessage = 'You requested to predict IMDB Score for the Movie '+str(input1)+' ....... '
        returnmessage = returnmessage +' and the score is '+str(imdb_score)+ 'View the Explainability of the prediction in the Model Explainability Tab'

        print('Call to Predict Completed .......')
        return returnmessage




if __name__ == '__main__':
    app.run_server(debug=True)
