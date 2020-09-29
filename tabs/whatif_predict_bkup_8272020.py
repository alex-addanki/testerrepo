import dash
from dash.dependencies import Input, Output , State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from app import app
import time
import pickle
import dill
import lime
import numpy as np

import lime
import lime.lime_tabular
from sklearn.externals import joblib

import pdfcrowd
import sys
import base64
import os
from model.whatif_1_dataprep import wif1_preparefeatures

###filename_VAG='model/AUG25_VAG_RFRegModel__24082020_234939.h5'

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
from images import avatar

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
from tabs import explain,explain_whatif2

####from model.makegauge import creategauge
from model.makegauge_1 import creategauge_1

filename='model\DecisionTreeRegressorModel_IMDB18082020_022901.h5'
##       'model\DecisionTreeRegressorModel_IMDB18082020_022901.h5'

file = open('data\DEMO_ALL_Features_AUG25', 'rb')
all5000_All = pickle.load(file)
file.close()

print('all5000_All.columns =',all5000_All.columns)

df = all5000_All[['original_title', 'Director_1','Actor_1','Actor_2','budget']]

###movie_titles=df['original_title'].unique()
###movie_titles = np.sort(movie_titles).tolist()

movie_titles = cnfg.Display_Movie_List

pred_report = pd.DataFrame(columns = ['Label','Value'],index = ['7'])




layout = dcc.Loading(html.Div([
    html.Button(id='show-whatifbutton-state', n_clicks=0, children=''),
    html.Button(id='whatifexplain-button', n_clicks=0, children='Explain Model'),

    html.Hr(),

    dash_table.DataTable(
        id='table-whatiffiltered-data',
        columns=[
            {"name": i, "id": i} for i in ['Label', 'Value']  ##sorted(df.columns)###df.columns
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'height': 'auto',
            # all three widths are needed
            'minWidth': '100px',
            'width': '100px',
            'maxWidth': '100px',
            'whiteSpace': 'normal'
        },
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'center'
            } for c in ['Labels', 'Values']
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    ),
    html.Div(id='display-whatifpredicted-values'),
    html.Div(id='display-whatifpredicted-graphs'),

]) , type='cube', fullscreen=True)


@app.callback(
    dash.dependencies.Output('table-whatiffiltered-data', 'data'),
    [Input('show-whatifbutton-state', 'n_clicks')]
    )
def show_whatifpredict_children(n_clicks):
    if n_clicks >= 0:
        #---
        unpickled_df = pd.read_pickle("data/updatedmovielist.pkl")
        print('UNPICKLED')
        print('unpickled_df',unpickled_df.columns)
        print(type(unpickled_df))

        modified_list=unpickled_df['ModifiedValues'].tolist()
        actuals_list = unpickled_df['Actual'].tolist()

        title = modified_list[0]
        print('VALUES-1', title)
        if is_nan(title):
            title = actuals_list[0]

        director = modified_list[1]
        print('VALUES-2',director)
        if is_nan(director):
            director = actuals_list[1]

        actor1=modified_list[2]
        print('VALUES-3', actor1)
        if is_nan(actor1):
            actor1 = actuals_list[2]

        actor2 = modified_list[3]
        print('VALUES-4', actor2)
        if is_nan(actor2):
            actor2 = actuals_list[3]

        budget = actuals_list[4] ###modified_list[4]
        print('VALUES-5', budget)
        if is_nan(budget):
            budget = actuals_list[4]
        print('inside NEW WHATIF LOUT with VALUES',title,director,actor1,actor2,budget)



        x777, explainer = wif1_preparefeatures(title, director, actor1, actor2, budget)
        print('INPUT FEATURE SIZE = ', x777.shape)

        ###LOAD MODEL PICKLE FOR PREDICTIONS

        loaded_model_VAG7 = pickle.load(
            open("model/whatif_models/RandomForrestRegressorModel_VAVG_21082020_213610.h5", "rb"))
        print('#######------->pkl file loaded', loaded_model_VAG7)

        ###PREDICT

        result = loaded_model_VAG7.predict(x777.reshape(1, -1))
        print('THE IMDB SCORE PRDICTED IS = ', str(result))

        ###LIME EXPLAINER

        np.random.seed(42)
        exp = explainer.explain_instance(x777, loaded_model_VAG7.predict, num_features=8)
        print('DELETING OLDER HTML FILE')
        ###os.remove("model/oi.html")
        exp.save_to_file('model/oi.html')
        print('Explainer Saved')

        try:
            # create the API client instance
            client = pdfcrowd.HtmlToImageClient('demo', 'ce544b6ea52a5621fb9d55f8b542d14d')

            # configure the conversion
            client.setOutputFormat('png')
            print('########################older pngs deleting')
            try:
                os.remove("reports/VAG_LIME_EXPLAINER.png")
            except:
                prin('file not there to delete')
            print('########################older pngs DELETED')
            # run the conversion and write the result to a file
            client.convertFileToFile('model/oi.html', 'reports/VAG_LIME_EXPLAINER.png')

            ###lime_png    = 'model/VAG_LIME_EXPLAINER.png'
            ###lime_base64 = base64.b64encode(open(lime_png, 'rb').read()).decode('ascii')
            print('LIME SAVED !!!!!!!')

        except pdfcrowd.Error as why:
            # report the error
            sys.stderr.write('Pdfcrowd Error: {}\n'.format(why))

            # rethrow or handle the exception
            raise
        pred_report = pd.DataFrame(columns=['Label', 'Value'], index=['7'])
        targetval= result[0] * 10
        pred_report = pred_report.append({'Label':'Predicted IMDB SCORE','Value':str(targetval)}, ignore_index=True)

        movie_idselctd = all5000_All[all5000_All['original_title'] == title].index.values[0]
        movie_iimdbactual = all5000_All["vote_average"].iloc[movie_idselctd]
        pred_report = pred_report.append({'Label': 'Actual IMDB SCORE', 'Value': str(movie_iimdbactual)}, ignore_index=True)
        predicteddatatable = pred_report.to_dict('records')


        value_actual = int(movie_iimdbactual) * 10
        print('creategauge calling with value_actual = ',value_actual)
        creategauge_1(value_actual,'reports\imdm_actual')
        print('##########creategauge calling Done')

        value_predicted = int(targetval) * 10
        print('creategauge calling with value_predicted = ', value_predicted)
        creategauge_1(value_predicted,'reports\imdm_predicted')
        print('creategauge done for predicted value')

    return predicteddatatable

@app.callback(Output('display-whatifpredicted-graphs', 'children'),
              [Input('whatifexplain-button', 'n_clicks')]
              )
def show_whatifexplanation77(n_clicks):
    if n_clicks > 0:
        print('explain_whatif----CLICKED')
        return explain_whatif2.layout
    else:
        return ''



def is_nan(x):
    return (x != x)

if __name__ == '__main__':
    app.run_server(debug=True)