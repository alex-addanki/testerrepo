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


df= filepreprocess.preprocess()
movie_titles=df['original_title'].unique().tolist()
Directors=df['Director_1'].unique().tolist()
Directors = np.sort(Directors).tolist()

actors=df['Actor_1'].unique().tolist()
actors = np.sort(actors).tolist()


cities=Directors

prefill = df[df['original_title'] == 'Tangled']
print('-------------------->>>',prefill.original_title)
print('-------------------->>>',prefill.revenue)

style = {'padding': '1.5em'}

movie=str(prefill.original_title)

layout = html.Div([
    dcc.Markdown("""
        ### Dream-Team Movie

    """),


    html.Div([
        dcc.Markdown('###### Movie Name'),
        dcc.Input(
            id='movie-input',
            style={'width': 300},
            value='Name of the Movie Here'
        ),
    ], style=style),

    ##html.Div([
    ##    dcc.Markdown('###### Movie Name'),
    ##    dcc.Dropdown(
    ##        id='movie-dropdown',
    ##        options=[{'label': k, 'value': k} for k in movie_titles],
    ##        value='Tangled'
    ##    ),
    ##], style=style),

    html.Div([
        dcc.Markdown('###### Budget'),
        dcc.Slider(
            id='budget',
            min  =1000000,
            step =1000000,
            max  =10000000,
            value=1500000,
            marks={n: f'{n / 1000:.0f}k' for n in range(1000000, 10000000, 1000000)}
        ),
    ], style=style),

    html.Div([
        dcc.Markdown('###### Director'),
        dcc.Dropdown(
            id='director-name',
            options=[{'label': director, 'value': director} for director in Directors],
            value=Directors[7]
        ),
    ], style=style),

    ###html.Div([
    ###    dcc.Markdown('###### Director - 2'),
    ###    dcc.Dropdown(
    ###        id='area',
    ###        options=[{'label': city, 'value': city} for city in cities],
    ###        value=cities[10]
    ###   ),
    ###], style=style),


    html.Div([
        dcc.Markdown('###### Actor - 1'),
        dcc.Dropdown(
            id='actor1-name',
            options=[{'label': actor1, 'value': actor1} for actor1 in actors],
            value=actors[7]
        ),
    ], style=style),

    html.Div([
        dcc.Markdown('###### Actor - 2'),
        dcc.Dropdown(
            id='actor2-name',
            options=[{'label': actor2, 'value': actor2} for actor2 in actors],
            value=actors[17]
        ),
    ], style=style),

    html.Button(id='predict-button-state', n_clicks=0, children='Predict-IMDB-Score'),

    html.Div(id='prediction-content', style={'fontWeight': 'bold'}),

])


@app.callback(
    Output('prediction-content', 'children'),
    [Input('movie-input', 'value'),
     Input('budget', 'value'),
     Input('director-name', 'value'),
     Input('actor1-name', 'value'),
     Input('actor2-name', 'value'),
     Input('predict-button-state', 'n_clicks')])
def predict(moviename, budget, directorname, actor1, actor2,n_clicks):
    dreamdf = pd.DataFrame(
        columns=['original_title', 'budget', 'Director_1', 'Actor_1', 'Actor_2'],
        data=[[moviename, budget, directorname, actor1, actor2]]
    )
    modified_values = 'Movie-Title = ' + str(dreamdf.original_title[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Director = ' + str(dreamdf.Director_1[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Actor-1 = ' + str(dreamdf.Actor_1[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Actor-2 = ' + str(dreamdf.Actor_2[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Budget = ' + str(dreamdf.budget[0])
    print(modified_values)
    dreamdf.to_pickle("data/dreammovie.pkl")
    print('Data pkld')
    return modified_values
