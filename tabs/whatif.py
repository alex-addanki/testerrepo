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
from model.makegauge import creategauge
filename_VAG='model/AUG25_VAG_RFRegModel__24082020_234939.h5'

###app = dash.Dash(__name__)

G_Title_value = ''
G_Director_value = ''
G_Actor_1_value = ''
G_Actor_2_value = ''
G_Budget_value = ''

###lime_png    = 'model/VAG_LIME_EXPLAINER.png'
###lime_base64 = base64.b64encode(open(lime_png, 'rb').read()).decode('ascii')

value_store = pd.DataFrame(columns = ['original_title', 'director','actor_1','actor_2','budget'],index = ['7'])

all_options = {
                'Tangled': {
                            'Director'     : ['Byron Howard', 'Nathan Greno', 'Ron Perlman'],
                            'Actor-1': ['Zachary Levi', 'Mandy Moore', 'Ron Perlman'],
                            'Actor-2': ['Donna Murphy', 'Rick Moore', 'Will Smith'],
                            'Budget': ['260000k', '350000k', '500000k'],
                            },
                'The Princess Diaries': {
                            'Director'     : ['Garry Marshall', 'Byron Howard', 'Nathan Greno'],
                            'Actor-1': ['Anne Hathaway', 'Julie Andrews', 'Heather Matarazzo'],
                            'Actor-2': ['Héctor Elizondo', 'John Rhys-Davies', 'Heather Matarazzo'],
                            'Budget': ['37000k', '50000k', '75000k'],
                            },
                'Avengers: Age of Ultron': {
                    'Director': ['Joss Whedon','Trudy Ramirez','Garry Marshall', 'Byron Howard', 'Nathan Greno'],
                    'Actor-1': ['Robert Downey Jr.', 'Julie Andrews', 'Heather Matarazzo'],
                    'Actor-2': ['Chris Hemsworth', 'John Rhys-Davies', 'Heather Matarazzo'],
                    'Budget': ['280000k', '350000k', '500000k'],
                }
}


layout = dcc.Loading(html.Div([
    dcc.Markdown('###### Select the Movie'),
    dcc.Dropdown(
        id='countries-dropdown',
        options=[{'label': k, 'value': k} for k in all_options.keys()],
        value='Tangled'
    ),

    html.Hr(),

    dcc.Dropdown(id='cities-dropdown'),

    html.Hr(),

    dcc.Dropdown(id='landmarks-dropdown'),

    html.Button(id='modify-button-state', n_clicks=0, children='Modify'),
    html.Button(id='predict-button-state', n_clicks=0, children='Predict'),

    html.Div(id='display-selected-values'),

    html.Div(id='display-predicted-title-values'),
    html.Div(id='display-predicted-director-values'),
    html.Div(id='display-predicted-actor-1-values'),
    html.Div(id='display-predicted-actor-2-values'),
    html.Div(id='display-predicted-budget-values'),
    html.Div(id='display-predicted-values'),
    ###html.Img(src='data:image/png;base64,{}'.format(lime_base64)),

]) , type='cube', fullscreen=True)


@app.callback(
    dash.dependencies.Output('cities-dropdown', 'options'),
    [dash.dependencies.Input('countries-dropdown', 'value')])
def set_cities_options(selected_country):
    return [{'label': i, 'value': i} for i in all_options[selected_country]]


@app.callback(
    dash.dependencies.Output('cities-dropdown', 'value'),
    [dash.dependencies.Input('cities-dropdown', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']

@app.callback(
    dash.dependencies.Output('landmarks-dropdown', 'options'),
    [dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cities-dropdown', 'value')])
def set_landmarks_options(selected_country, selected_city):
    return [{'label': i, 'value': i} for i in all_options[selected_country][selected_city]]


@app.callback(
    dash.dependencies.Output('landmarks-dropdown', 'value'),
    [dash.dependencies.Input('landmarks-dropdown', 'options')])
def set_landmarks_value(available_options):
    return available_options[0]['value']


@app.callback([
    Output('display-selected-values', 'children'),
    Output('display-predicted-title-values', 'children'),  ###title
    Output('display-predicted-director-values', 'children'),  ###director
    Output('display-predicted-actor-1-values', 'children'),  ###actor-1
    Output('display-predicted-actor-2-values', 'children'),  ###actor-2
    Output('display-predicted-budget-values', 'children')  ###budget
    ],
    [Input('modify-button-state', 'n_clicks'),
     dash.dependencies.Input('countries-dropdown', 'value'),
     dash.dependencies.Input('cities-dropdown', 'value'),
     dash.dependencies.Input('landmarks-dropdown', 'value')
     ]
    )
def set_display_children(n_clicks,selected_country, selected_city, selected_landmark):
    ###time.sleep(1)
    print('Number of clicks = ',n_clicks)
    if n_clicks == 0 : ###not clicked
        Title_value    = ''
        Director_value = ''
        Actor_1_value  = ''
        Actor_2_value  = ''
        Budget_value   = ''

        Final_value = Title_value + Director_value + Actor_1_value  + Actor_2_value  + Budget_value

        print('not clicked - Final_value = ',Final_value)
    elif n_clicks == 1 : ###clicked update for the first time
        print('INPUT VALUES-->',selected_country, selected_city, selected_landmark)
        Title_value='Movie-Title = '+selected_country
        ###G_Title_value = selected_country
        value_store.at['7', 'original_title'] = str(selected_country)
        if selected_city == 'Director':
            print('Director Selected')
            Director_value = 'Director = ' + selected_landmark
            value_store.at['7', 'director'] = str(selected_landmark)
        if selected_city == 'Actor-1':
            print('Actor-1 Selected')
            Actor_1_value = 'Actor-1='+selected_landmark
            value_store.at['7', 'actor_1'] = str(selected_landmark)
        if selected_city == 'Actor-2':
            print('Actor-2 Selected')
            Actor_2_value = 'Actor-2='+selected_landmark
            value_store.at['7', 'actor_2'] = str(selected_landmark)
        if selected_city == 'Budget':
            print('Budget Selected')
            Budget_value = 'Budget='+selected_landmark
            value_store.at['7', 'budget'] = str(selected_landmark)
        ###Final_value = Title_value + Director_value + Actor_1_value + Actor_2_value + Budget_value
    elif n_clicks > 1:
        print('INPUT VALUES-->',selected_country, selected_city, selected_landmark)
        Title_value='Movie-Title = '+selected_country
        value_store.at['7', 'original_title'] = str(selected_country)
        if selected_city == 'Director':
            print('Director Selected')
            Director_value = 'Director = ' + selected_landmark
            value_store.at['7', 'director'] = str(selected_landmark)
        if selected_city == 'Actor-1':
            print('Actor-1 Selected')
            Actor_1_value = 'Actor-1='+selected_landmark
            value_store.at['7', 'actor_1'] = str(selected_landmark)
        if selected_city == 'Actor-2':
            print('Actor-2 Selected')
            Actor_2_value = 'Actor-2='+selected_landmark
            value_store.at['7', 'actor_2'] = str(selected_landmark)
        if selected_city == 'Budget':
            print('Budget Selected')
            Budget_value = 'Budget='+selected_landmark
            value_store.at['7', 'budget'] = str(selected_landmark)
        ###Final_value = Title_value + Director_value + Actor_1_value + Actor_2_value + Budget_value
    print(value_store.original_title)
    modified_values = 'Movie-Title = '+str(value_store.original_title[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values +'Director = ' + str(value_store.director[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Actor-1 = ' + str(value_store.actor_1[0])
    modified_values = modified_values +  '-------\n'
    modified_values = modified_values + 'Actor-2 = ' + str(value_store.actor_2[0])
    modified_values = modified_values +  '-------\n'
    modified_values = modified_values + 'Budget = ' + str(value_store.budget[0])
    print(modified_values)
    value_store.to_pickle("data/modifymovie.pkl")
    ##modified_values=value_store.values.tolist()
    ###modified_values='Updated Values are'+modified_values
    ##return u'Movie - {} ''s {} is now changed to {} '.format(selected_country, selected_city, selected_landmark, )

    print('--------------->>>',G_Title_value)
    ###set_predict_children(1,G_Title_value,G_Title_value,G_Title_value)
    return modified_values,\
           value_store.original_title[0],\
           value_store.director[0],\
           value_store.actor_1[0],\
           value_store.actor_2[0],\
           value_store.budget[0]


@app.callback(
    dash.dependencies.Output('display-predicted-values', 'children'),
    [Input('predict-button-state', 'n_clicks'),
     dash.dependencies.Input('display-predicted-title-values', 'children'),
     dash.dependencies.Input('display-predicted-director-values', 'children'),
     dash.dependencies.Input('display-predicted-actor-1-values', 'children'),
     dash.dependencies.Input('display-predicted-actor-2-values', 'children'),
     dash.dependencies.Input('display-predicted-budget-values', 'children'),
     ]
    )
def set_predict_children7(n_clicks, title,director,actor1,actor2,budget):
    print('Number of clicks = ',n_clicks)
    if n_clicks > 0 : ###not clicked
        ####time.sleep(3)
        ##print('set_predict_children--------------->>>',title,director,actor1,actor2,budget)

        ###LOAD MODEL for PREDICTION
        x777,explainer = wif1_preparefeatures(title,director,actor1,actor2,budget)
        print('INPUT FEATURE SIZE = ',x777.shape)
        loaded_model_VAG7 = pickle.load(open("model/whatif_models/RandomForrestRegressorModel_VAVG_21082020_213610.h5", "rb"))
        print('#######------->pkl file loaded',loaded_model_VAG7)
        result = loaded_model_VAG7.predict(x777.reshape(1, -1))
        print('THE IMDB SCORE PRDICTED IS = ',str(result))

        ###LIME EXPLAINER
        np.random.seed(42)
        exp = explainer.explain_instance(x777, loaded_model_VAG7.predict, num_features=8)
        ###exp.show_in_notebook(show_all=False)  # only the features used in the explanation are displayed
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




        return 'The Predicted Score for the Crew ,Cast and Budget Combination is --'+str(result[0])


if __name__ == '__main__':
    app.run_server(debug=True)