from dash.dependencies import Input, Output , State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import time
from joblib import load
import numpy as np
import pandas as pd
from app import app
from model import filepreprocess_demo
from model.prediction_explainability import load_prediction_explainability_metrics
import Config as cnfg


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

from tabs import viewall
from images import show_poster_speedo_avatar,show_poster_speedo_tangled,show_poster_speedo_loveletter,sps_mov4,sps_mov5,sps_mov6,sps_mov7

import plotly.graph_objects as go

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
import flask
import base64
from tabs import explain
from model.makegauge_1 import creategauge_1

import PIL
from PIL import Image


filename='model\DecisionTreeRegressorModel_IMDB18082020_022901.h5'
df= filepreprocess_demo.preprocess_demo()

###movie_titles=df['original_title'].unique()
###movie_titles = np.sort(movie_titles).tolist()

movie_titles = cnfg.Display_Movie_List


image_directory = 'images'
###list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

poster = base64.b64encode(open('images/Avatar.png', 'rb').read()).decode('ascii')

###style = {'padding': '1.5em'}


layout = dcc.Loading(
    html.Div([
    dcc.Markdown('###### Select a Movie'),
    dcc.Dropdown(
        id='movie-dropdown',
        options=[{'label': k, 'value': k} for k in movie_titles],
        value='Movie-Title'
    ),

    html.Hr(),

    html.Button(id='submit-button-state', n_clicks=0, children='Get Me IMDB Predictions'),
    html.Button(id='submit-button-state-2', n_clicks=0, children='Show Me Poster'),
    html.Button(id='submit-button-state-3', n_clicks=0, children='Explain Me Model'),
    html.Div(id='output-state'),
    html.Div(id='output-state-2'),
    dash_table.DataTable(
                        id='table-filtering-be',
                        columns=[
                            {"name": i, "id": i} for i in ['Labels','Values'] ##sorted(df.columns)###df.columns
                         ],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                                'height': 'auto',
                                 # all three widths are needed
                                'minWidth': '180px',
                                'width': '250px',
                                'maxWidth': '250px',
                                'whiteSpace': 'normal'
                            },
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'center'
                            } for c in ['Labels','Values']
                        ],
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold' ,
                            'column-header-name': '-----------'
                        }
                    ),


                   ###dcc.Graph(id='speedometer'),
                   html.Div(id='output-state-speedo'),
                   ###html.Div(id='output-state-2'),
                   html.Div(id='output-state-3'),
]) , type='cube', fullscreen=False)


@app.callback(Output('table-filtering-be', 'data'),
              [Input('submit-button-state', 'n_clicks')],
              [State('movie-dropdown', 'value')])
def update_output_intro(n_clicks, input1):
    figure = ''
    if n_clicks > 0:
        x=df[df['original_title'] == input1]
        intro_data = x.T
        intro_data = intro_data.reset_index(drop=False)
        print('Index reset')
        intro_data.rename(columns={intro_data.columns[0]: "Labels"}, inplace=True)
        intro_data.rename(columns={intro_data.columns[1]: "Values"}, inplace=True)
        print('columns renamed')

        ##############################################################################
        ##################PREDICTIONS#################################################
        ##############################################################################
        validation_df = input1
        preprocessed_data = preparefeatures(validation_df)
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

        intro_data = intro_data.append({'Labels':'Predicted IMDB SCORE','Values':str(imdb_score)}, ignore_index=True)

        print('Call to Predict Completed .......')

        datatable = intro_data.to_dict('records')

        # figure = go.Figure(go.Indicator(
        #     mode="gauge+number",
        #     value=imdb_score,
        #     domain={'x': [0, 1], 'y': [0, 1]},
        #     title={'text': "IMDB-VoteAverage"}))
        # figure.show()
        value = int(imdb_score) * 10
        creategauge_1(value,'images\predicted')
        print('GAUGE COMPLETED',value)
        avatar_png = 'images\Avatar.png'
        image1 = Image.open(avatar_png)
        WIDTH, HEIGHT = image1.size
        print(f'POSTER ----- {WIDTH} * {HEIGHT}')

        avatar_speedo_png = 'images/predicted.png'
        image = Image.open(avatar_speedo_png)
        WIDTH, HEIGHT = image.size
        print(f'speedo ----- {WIDTH} * {HEIGHT}')


        return datatable

@app.callback(Output('output-state-2', 'children'),
    [Input('submit-button-state-2', 'n_clicks')],
    [State('movie-dropdown', 'value')])
def update_poster_src(n_clicks,value):
    if value == 'Avatar' and n_clicks > 0:
        return show_poster_speedo_avatar.layout
    elif value == 'Tangled' and n_clicks > 0:
        return show_poster_speedo_tangled.layout
    elif value == 'Love Letters' and n_clicks > 0:
        return show_poster_speedo_loveletter.layout
    elif value == 'The Dark Knight Rises' and n_clicks > 0:
        return sps_mov4.layout
    elif value == 'John Carter' and n_clicks > 0:
        return sps_mov5.layout
    elif value == 'Avengers: Age of Ultron' and n_clicks > 0:
        return sps_mov6.layout
    elif value == 'The Avengers' and n_clicks > 0:
        return sps_mov7.layout

@app.callback(Output('output-state-3', 'children'),
    [Input('submit-button-state-3', 'n_clicks')],
    [State('movie-dropdown', 'value')])
def show_explanation(n_clicks,value):
    if n_clicks > 0:
        return explain.layout
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)