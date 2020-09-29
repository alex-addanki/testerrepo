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

filename_VAG='model/AUG25_VAG_RFRegModel__24082020_234939.h5'

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
from tabs import explain,whatif_predict


file = open('data\DEMO_ALL_Features_AUG25', 'rb')
all5000_All = pickle.load(file)
file.close()

print('all5000_All.columns =',all5000_All.columns)

df = all5000_All[['original_title', 'Director_1','Actor_1','Actor_2','budget']]

###movie_titles=df['original_title'].unique()
###movie_titles = np.sort(movie_titles).tolist()

movie_titles = cnfg.Display_Movie_List

value_store = pd.DataFrame(columns = ['original_title', 'Director_1','Actor_1','Actor_2','budget'],index = ['7'])

all_options = {
                'Tangled': {
                            'Director'     : ['Byron Howard', 'Martin Scorsese','Steven Spielberg','Christopher Nolan','Quentin Tarantino','Clint Eastwood','David Fincher','Peter Jackson','Woody Allen','Francis Ford Coppola','Stanley Kubrick','Robert Zemeckis','Wes Anderson','Richard Linklater','James Cameron','Hayao Miyazaki','Ridley Scott','Sergio Leone','Tim Burton','Alfred Hitchcock','Joel Coen','Danny Boyle','Paul Thomas Anderson','Billy Wilder','Oliver Stone','Ron Howard'],
                            'Actor-1': ['Zachary Levi', 'Tom Hanks','Leonardo DiCaprio','Denzel Washington','Robert De Niro','Christian Bale','Johnny Depp','Brad Pitt','Clint Eastwood','Al Pacino','Tom Cruise','Daniel Radcliffe','Paul Newman','Matt Damon','Edward Norton','Kevin Spacey','Robert Downey Jr.','Jeff Bridges','Jack Nicholson','Sean Penn','Russell Crowe','Hugh Jackman','Mark Hamill','Mel Gibson','Kevin Costner','Woody Allen','Will Smith','Anthony Hopkins','Joaquin Phoenix','Kate Winslet','Gregory Peck','Ralph Fiennes','Cate Blanchett','Jamie Foxx','Kurt Russell'],
                            'Actor-2': ['Mandy Moore', 'Jennifer Aniston','Morgan Freeman','Brad Pitt','Samuel L. Jackson','Alec Baldwin','Scarlett Johansson','Robert De Niro','Diane Keaton','Kate Winslet','Matt Damon','Gene Hackman','Julianne Moore','Josh Hutcherson','Gary Oldman','Philip Seymour Hoffman','Christopher Plummer','Ewan McGregor','Colin Firth','Cate Blanchett','Laurence Fishburne','Ben Kingsley','Charlize Theron','Gwyneth Paltrow','Drew Barrymore','Dustin Hoffman','Aaron Eckhart','Susan Sarandon','Danny DeVito','Justin Long','Natalie Portman','Jude Law','Anthony Hopkins','Tommy Lee Jones','Annette Bening','Rose Byrne','Zoe Saldana','Mark Ruffalo','Michelle Pfeiffer','Gabriel Byrne','Ashley Judd','Angelina Jolie','Julia Roberts','Emma Stone','Salma Hayek','Owen Wilson','Rachel McAdams','Robert Duvall','Anne Hathaway','Rachel Weisz','Robert Downey Jr.','Hugh Jackman','Meryl Streep','Catherine Keener','Willem Dafoe','Channing Tatum','Maria Bello','Matthew McConaughey','Winona Ryder','Emily Blunt','Reese Witherspoon','Dennis Quaid','Nicole Kidman','Renée Zellweger','Mark Wahlberg','Greg Kinnear','Woody Harrelson','Catherine Zeta-Jones','Billy Bob Thornton','John Malkovich','Sean Penn','Liam Neeson','Kate Hudson','Jessica Biel','Rupert Grint','James McAvoy','Helen Hunt','Jake Gyllenhaal','Michael Fassbender','Gerard Butler','Keira Knightley','John Goodman','Kevin Spacey','Michael Caine','Patricia Arquette','Cuba Gooding Jr.','Leonard Nimoy','Hugh Grant','Robin Williams','Russell Crowe','Forest Whitaker','Eddie Murphy','Danny Glover'],
                            'Budget': ['260000k', '350000k', '500000k'],
                            },
                'Avatar': {
                            'Director'     : ['James Cameron', 'Martin Scorsese','Steven Spielberg','Christopher Nolan','Quentin Tarantino','Clint Eastwood','David Fincher','Peter Jackson','Woody Allen','Francis Ford Coppola','Stanley Kubrick','Robert Zemeckis','Wes Anderson','Richard Linklater','James Cameron','Hayao Miyazaki','Ridley Scott','Sergio Leone','Tim Burton','Alfred Hitchcock','Joel Coen','Danny Boyle','Paul Thomas Anderson','Billy Wilder','Oliver Stone','Ron Howard'],
                            'Actor-1': ['Sam Worthington', 'Tom Hanks','Leonardo DiCaprio','Denzel Washington','Robert De Niro','Christian Bale','Johnny Depp','Brad Pitt','Clint Eastwood','Al Pacino','Tom Cruise','Daniel Radcliffe','Paul Newman','Matt Damon','Edward Norton','Kevin Spacey','Robert Downey Jr.','Jeff Bridges','Jack Nicholson','Sean Penn','Russell Crowe','Hugh Jackman','Mark Hamill','Mel Gibson','Kevin Costner','Woody Allen','Will Smith','Anthony Hopkins','Joaquin Phoenix','Kate Winslet','Gregory Peck','Ralph Fiennes','Cate Blanchett','Jamie Foxx','Kurt Russell'],
                            'Actor-2': ['Zoe Saldana', 'Jennifer Aniston','Morgan Freeman','Brad Pitt','Samuel L. Jackson','Alec Baldwin','Scarlett Johansson','Robert De Niro','Diane Keaton','Kate Winslet','Matt Damon','Gene Hackman','Julianne Moore','Josh Hutcherson','Gary Oldman','Philip Seymour Hoffman','Christopher Plummer','Ewan McGregor','Colin Firth','Cate Blanchett','Laurence Fishburne','Ben Kingsley','Charlize Theron','Gwyneth Paltrow','Drew Barrymore','Dustin Hoffman','Aaron Eckhart','Susan Sarandon','Danny DeVito','Justin Long','Natalie Portman','Jude Law','Anthony Hopkins','Tommy Lee Jones','Annette Bening','Rose Byrne','Zoe Saldana','Mark Ruffalo','Michelle Pfeiffer','Gabriel Byrne','Ashley Judd','Angelina Jolie','Julia Roberts','Emma Stone','Salma Hayek','Owen Wilson','Rachel McAdams','Robert Duvall','Anne Hathaway','Rachel Weisz','Robert Downey Jr.','Hugh Jackman','Meryl Streep','Catherine Keener','Willem Dafoe','Channing Tatum','Maria Bello','Matthew McConaughey','Winona Ryder','Emily Blunt','Reese Witherspoon','Dennis Quaid','Nicole Kidman','Renée Zellweger','Mark Wahlberg','Greg Kinnear','Woody Harrelson','Catherine Zeta-Jones','Billy Bob Thornton','John Malkovich','Sean Penn','Liam Neeson','Kate Hudson','Jessica Biel','Rupert Grint','James McAvoy','Helen Hunt','Jake Gyllenhaal','Michael Fassbender','Gerard Butler','Keira Knightley','John Goodman','Kevin Spacey','Michael Caine','Patricia Arquette','Cuba Gooding Jr.','Leonard Nimoy','Hugh Grant','Robin Williams','Russell Crowe','Forest Whitaker','Eddie Murphy','Danny Glover'],
                            'Budget': ['37000k', '50000k', '75000k'],
                            },
                'Love Letters': {
                            'Director'     : ['Amy Holden Jones', 'Martin Scorsese','Steven Spielberg','Christopher Nolan','Quentin Tarantino','Clint Eastwood','David Fincher','Peter Jackson','Woody Allen','Francis Ford Coppola','Stanley Kubrick','Robert Zemeckis','Wes Anderson','Richard Linklater','James Cameron','Hayao Miyazaki','Ridley Scott','Sergio Leone','Tim Burton','Alfred Hitchcock','Joel Coen','Danny Boyle','Paul Thomas Anderson','Billy Wilder','Oliver Stone','Ron Howard'],
                            'Actor-1': ['Jamie Lee Curtis', 'Tom Hanks','Leonardo DiCaprio','Denzel Washington','Robert De Niro','Christian Bale','Johnny Depp','Brad Pitt','Clint Eastwood','Al Pacino','Tom Cruise','Daniel Radcliffe','Paul Newman','Matt Damon','Edward Norton','Kevin Spacey','Robert Downey Jr.','Jeff Bridges','Jack Nicholson','Sean Penn','Russell Crowe','Hugh Jackman','Mark Hamill','Mel Gibson','Kevin Costner','Woody Allen','Will Smith','Anthony Hopkins','Joaquin Phoenix','Kate Winslet','Gregory Peck','Ralph Fiennes','Cate Blanchett','Jamie Foxx','Kurt Russell'],
                            'Actor-2': ['Bonnie Bartlett','Zoe Saldana', 'Jennifer Aniston','Morgan Freeman','Brad Pitt','Samuel L. Jackson','Alec Baldwin','Scarlett Johansson','Robert De Niro','Diane Keaton','Kate Winslet','Matt Damon','Gene Hackman','Julianne Moore','Josh Hutcherson','Gary Oldman','Philip Seymour Hoffman','Christopher Plummer','Ewan McGregor','Colin Firth','Cate Blanchett','Laurence Fishburne','Ben Kingsley','Charlize Theron','Gwyneth Paltrow','Drew Barrymore','Dustin Hoffman','Aaron Eckhart','Susan Sarandon','Danny DeVito','Justin Long','Natalie Portman','Jude Law','Anthony Hopkins','Tommy Lee Jones','Annette Bening','Rose Byrne','Zoe Saldana','Mark Ruffalo','Michelle Pfeiffer','Gabriel Byrne','Ashley Judd','Angelina Jolie','Julia Roberts','Emma Stone','Salma Hayek','Owen Wilson','Rachel McAdams','Robert Duvall','Anne Hathaway','Rachel Weisz','Robert Downey Jr.','Hugh Jackman','Meryl Streep','Catherine Keener','Willem Dafoe','Channing Tatum','Maria Bello','Matthew McConaughey','Winona Ryder','Emily Blunt','Reese Witherspoon','Dennis Quaid','Nicole Kidman','Renée Zellweger','Mark Wahlberg','Greg Kinnear','Woody Harrelson','Catherine Zeta-Jones','Billy Bob Thornton','John Malkovich','Sean Penn','Liam Neeson','Kate Hudson','Jessica Biel','Rupert Grint','James McAvoy','Helen Hunt','Jake Gyllenhaal','Michael Fassbender','Gerard Butler','Keira Knightley','John Goodman','Kevin Spacey','Michael Caine','Patricia Arquette','Cuba Gooding Jr.','Leonard Nimoy','Hugh Grant','Robin Williams','Russell Crowe','Forest Whitaker','Eddie Murphy','Danny Glover'],
                            'Budget': ['37000k', '50000k', '75000k'],
                            },
                'The Dark Knight Rises': {
                            'Director': ['Christopher Nolan','Amy Holden Jones', 'Martin Scorsese', 'Steven Spielberg', 'Christopher Nolan',
                                 'Quentin Tarantino', 'Clint Eastwood', 'David Fincher', 'Peter Jackson', 'Woody Allen',
                                 'Francis Ford Coppola', 'Stanley Kubrick', 'Robert Zemeckis', 'Wes Anderson', 'Richard Linklater',
                                 'James Cameron', 'Hayao Miyazaki', 'Ridley Scott', 'Sergio Leone', 'Tim Burton',
                                 'Alfred Hitchcock', 'Joel Coen', 'Danny Boyle', 'Paul Thomas Anderson', 'Billy Wilder',
                                 'Oliver Stone', 'Ron Howard'],
                            'Actor-1': ['Christian Bale','Jamie Lee Curtis', 'Tom Hanks', 'Leonardo DiCaprio', 'Denzel Washington', 'Robert De Niro',
                                 'Johnny Depp', 'Brad Pitt', 'Clint Eastwood', 'Al Pacino', 'Tom Cruise',
                                'Daniel Radcliffe', 'Paul Newman', 'Matt Damon', 'Edward Norton', 'Kevin Spacey',
                                'Robert Downey Jr.', 'Jeff Bridges', 'Jack Nicholson', 'Sean Penn', 'Russell Crowe', 'Hugh Jackman',
                                'Mark Hamill', 'Mel Gibson', 'Kevin Costner', 'Woody Allen', 'Will Smith', 'Anthony Hopkins',
                                'Joaquin Phoenix', 'Kate Winslet', 'Gregory Peck', 'Ralph Fiennes', 'Cate Blanchett', 'Jamie Foxx',
                                'Kurt Russell'],
                            'Actor-2': ['Michael Caine','Bonnie Bartlett', 'Zoe Saldana', 'Jennifer Aniston', 'Morgan Freeman', 'Brad Pitt',
                                'Samuel L. Jackson', 'Alec Baldwin', 'Scarlett Johansson', 'Robert De Niro', 'Diane Keaton',
                                'Kate Winslet', 'Matt Damon', 'Gene Hackman', 'Julianne Moore', 'Josh Hutcherson', 'Gary Oldman',
                                'Philip Seymour Hoffman', 'Christopher Plummer', 'Ewan McGregor', 'Colin Firth', 'Cate Blanchett',
                                'Laurence Fishburne', 'Ben Kingsley', 'Charlize Theron', 'Gwyneth Paltrow', 'Drew Barrymore',
                                'Dustin Hoffman', 'Aaron Eckhart', 'Susan Sarandon', 'Danny DeVito', 'Justin Long',
                                'Natalie Portman', 'Jude Law', 'Anthony Hopkins', 'Tommy Lee Jones', 'Annette Bening', 'Rose Byrne',
                                'Zoe Saldana', 'Mark Ruffalo', 'Michelle Pfeiffer', 'Gabriel Byrne', 'Ashley Judd',
                                'Angelina Jolie', 'Julia Roberts', 'Emma Stone', 'Salma Hayek', 'Owen Wilson', 'Rachel McAdams',
                                'Robert Duvall', 'Anne Hathaway', 'Rachel Weisz', 'Robert Downey Jr.', 'Hugh Jackman',
                                'Meryl Streep', 'Catherine Keener', 'Willem Dafoe', 'Channing Tatum', 'Maria Bello',
                                'Matthew McConaughey', 'Winona Ryder', 'Emily Blunt', 'Reese Witherspoon', 'Dennis Quaid',
                                'Nicole Kidman', 'Renée Zellweger', 'Mark Wahlberg', 'Greg Kinnear', 'Woody Harrelson',
                                'Catherine Zeta-Jones', 'Billy Bob Thornton', 'John Malkovich', 'Sean Penn', 'Liam Neeson',
                                'Kate Hudson', 'Jessica Biel', 'Rupert Grint', 'James McAvoy', 'Helen Hunt', 'Jake Gyllenhaal',
                                'Michael Fassbender', 'Gerard Butler', 'Keira Knightley', 'John Goodman', 'Kevin Spacey',
                                'Michael Caine', 'Patricia Arquette', 'Cuba Gooding Jr.', 'Leonard Nimoy', 'Hugh Grant',
                                'Robin Williams', 'Russell Crowe', 'Forest Whitaker', 'Eddie Murphy', 'Danny Glover'],
                            'Budget': ['37000k', '50000k', '75000k']
                            },
                'Avengers: Age of Ultron': {
                    'Director': ['Joss Whedon','Andrew Stanton', 'Amy Holden Jones', 'Martin Scorsese', 'Steven Spielberg',
                                 'Christopher Nolan',
                                 'Quentin Tarantino', 'Clint Eastwood', 'David Fincher', 'Peter Jackson', 'Woody Allen',
                                 'Francis Ford Coppola', 'Stanley Kubrick', 'Robert Zemeckis', 'Wes Anderson',
                                 'Richard Linklater',
                                 'James Cameron', 'Hayao Miyazaki', 'Ridley Scott', 'Sergio Leone', 'Tim Burton',
                                 'Alfred Hitchcock', 'Joel Coen', 'Danny Boyle', 'Paul Thomas Anderson', 'Billy Wilder',
                                 'Oliver Stone', 'Ron Howard'],
                    'Actor-1': ['Robert Downey Jr.', 'Taylor Kitsch','Christian Bale', 'Jamie Lee Curtis', 'Tom Hanks', 'Leonardo DiCaprio', 'Denzel Washington',
                                'Robert De Niro',
                                'Johnny Depp', 'Brad Pitt', 'Clint Eastwood', 'Al Pacino', 'Tom Cruise',
                                'Daniel Radcliffe', 'Paul Newman', 'Matt Damon', 'Edward Norton', 'Kevin Spacey',
                                'Jeff Bridges', 'Jack Nicholson', 'Sean Penn', 'Russell Crowe',
                                'Hugh Jackman',
                                'Mark Hamill', 'Mel Gibson', 'Kevin Costner', 'Woody Allen', 'Will Smith', 'Anthony Hopkins',
                                'Joaquin Phoenix', 'Kate Winslet', 'Gregory Peck', 'Ralph Fiennes', 'Cate Blanchett',
                                'Jamie Foxx',
                                'Kurt Russell'],
                    'Actor-2': ['Chris Hemsworth','Lynn Collins','Michael Caine', 'Bonnie Bartlett', 'Zoe Saldana', 'Jennifer Aniston', 'Morgan Freeman',
                                'Brad Pitt',
                                'Samuel L. Jackson', 'Alec Baldwin', 'Scarlett Johansson', 'Robert De Niro', 'Diane Keaton',
                                'Kate Winslet', 'Matt Damon', 'Gene Hackman', 'Julianne Moore', 'Josh Hutcherson',
                                'Gary Oldman',
                                'Philip Seymour Hoffman', 'Christopher Plummer', 'Ewan McGregor', 'Colin Firth',
                                'Cate Blanchett',
                                'Laurence Fishburne', 'Ben Kingsley', 'Charlize Theron', 'Gwyneth Paltrow', 'Drew Barrymore',
                                'Dustin Hoffman', 'Aaron Eckhart', 'Susan Sarandon', 'Danny DeVito', 'Justin Long',
                                'Natalie Portman', 'Jude Law', 'Anthony Hopkins', 'Tommy Lee Jones', 'Annette Bening',
                                'Rose Byrne',
                                'Zoe Saldana', 'Mark Ruffalo', 'Michelle Pfeiffer', 'Gabriel Byrne', 'Ashley Judd',
                                'Angelina Jolie', 'Julia Roberts', 'Emma Stone', 'Salma Hayek', 'Owen Wilson', 'Rachel McAdams',
                                'Robert Duvall', 'Anne Hathaway', 'Rachel Weisz', 'Robert Downey Jr.', 'Hugh Jackman',
                                'Meryl Streep', 'Catherine Keener', 'Willem Dafoe', 'Channing Tatum', 'Maria Bello',
                                'Matthew McConaughey', 'Winona Ryder', 'Emily Blunt', 'Reese Witherspoon', 'Dennis Quaid',
                                'Nicole Kidman', 'Renée Zellweger', 'Mark Wahlberg', 'Greg Kinnear', 'Woody Harrelson',
                                'Catherine Zeta-Jones', 'Billy Bob Thornton', 'John Malkovich', 'Sean Penn', 'Liam Neeson',
                                'Kate Hudson', 'Jessica Biel', 'Rupert Grint', 'James McAvoy', 'Helen Hunt', 'Jake Gyllenhaal',
                                'Michael Fassbender', 'Gerard Butler', 'Keira Knightley', 'John Goodman', 'Kevin Spacey',
                                'Michael Caine', 'Patricia Arquette', 'Cuba Gooding Jr.', 'Leonard Nimoy', 'Hugh Grant',
                                'Robin Williams', 'Russell Crowe', 'Forest Whitaker', 'Eddie Murphy', 'Danny Glover'],
                    'Budget': ['37000k', '50000k', '75000k']
                },

                'The Avengers': {
                    'Director': ['Joss Whedon','Andrew Stanton', 'Amy Holden Jones', 'Martin Scorsese', 'Steven Spielberg',
                                 'Christopher Nolan',
                                 'Quentin Tarantino', 'Clint Eastwood', 'David Fincher', 'Peter Jackson', 'Woody Allen',
                                 'Francis Ford Coppola', 'Stanley Kubrick', 'Robert Zemeckis', 'Wes Anderson',
                                 'Richard Linklater',
                                 'James Cameron', 'Hayao Miyazaki', 'Ridley Scott', 'Sergio Leone', 'Tim Burton',
                                 'Alfred Hitchcock', 'Joel Coen', 'Danny Boyle', 'Paul Thomas Anderson', 'Billy Wilder',
                                 'Oliver Stone', 'Ron Howard'],
                    'Actor-1': ['Robert Downey Jr.', 'Taylor Kitsch','Christian Bale', 'Jamie Lee Curtis', 'Tom Hanks', 'Leonardo DiCaprio', 'Denzel Washington',
                                'Robert De Niro',
                                'Johnny Depp', 'Brad Pitt', 'Clint Eastwood', 'Al Pacino', 'Tom Cruise',
                                'Daniel Radcliffe', 'Paul Newman', 'Matt Damon', 'Edward Norton', 'Kevin Spacey',
                                'Jeff Bridges', 'Jack Nicholson', 'Sean Penn', 'Russell Crowe',
                                'Hugh Jackman',
                                'Mark Hamill', 'Mel Gibson', 'Kevin Costner', 'Woody Allen', 'Will Smith', 'Anthony Hopkins',
                                'Joaquin Phoenix', 'Kate Winslet', 'Gregory Peck', 'Ralph Fiennes', 'Cate Blanchett',
                                'Jamie Foxx',
                                'Kurt Russell'],
                    'Actor-2': ['Chris Evans','Chris Hemsworth','Lynn Collins','Michael Caine', 'Bonnie Bartlett', 'Zoe Saldana', 'Jennifer Aniston', 'Morgan Freeman',
                                'Brad Pitt',
                                'Samuel L. Jackson', 'Alec Baldwin', 'Scarlett Johansson', 'Robert De Niro', 'Diane Keaton',
                                'Kate Winslet', 'Matt Damon', 'Gene Hackman', 'Julianne Moore', 'Josh Hutcherson',
                                'Gary Oldman',
                                'Philip Seymour Hoffman', 'Christopher Plummer', 'Ewan McGregor', 'Colin Firth',
                                'Cate Blanchett',
                                'Laurence Fishburne', 'Ben Kingsley', 'Charlize Theron', 'Gwyneth Paltrow', 'Drew Barrymore',
                                'Dustin Hoffman', 'Aaron Eckhart', 'Susan Sarandon', 'Danny DeVito', 'Justin Long',
                                'Natalie Portman', 'Jude Law', 'Anthony Hopkins', 'Tommy Lee Jones', 'Annette Bening',
                                'Rose Byrne',
                                'Zoe Saldana', 'Mark Ruffalo', 'Michelle Pfeiffer', 'Gabriel Byrne', 'Ashley Judd',
                                'Angelina Jolie', 'Julia Roberts', 'Emma Stone', 'Salma Hayek', 'Owen Wilson', 'Rachel McAdams',
                                'Robert Duvall', 'Anne Hathaway', 'Rachel Weisz', 'Robert Downey Jr.', 'Hugh Jackman',
                                'Meryl Streep', 'Catherine Keener', 'Willem Dafoe', 'Channing Tatum', 'Maria Bello',
                                'Matthew McConaughey', 'Winona Ryder', 'Emily Blunt', 'Reese Witherspoon', 'Dennis Quaid',
                                'Nicole Kidman', 'Renée Zellweger', 'Mark Wahlberg', 'Greg Kinnear', 'Woody Harrelson',
                                'Catherine Zeta-Jones', 'Billy Bob Thornton', 'John Malkovich', 'Sean Penn', 'Liam Neeson',
                                'Kate Hudson', 'Jessica Biel', 'Rupert Grint', 'James McAvoy', 'Helen Hunt', 'Jake Gyllenhaal',
                                'Michael Fassbender', 'Gerard Butler', 'Keira Knightley', 'John Goodman', 'Kevin Spacey',
                                'Michael Caine', 'Patricia Arquette', 'Cuba Gooding Jr.', 'Leonard Nimoy', 'Hugh Grant',
                                'Robin Williams', 'Russell Crowe', 'Forest Whitaker', 'Eddie Murphy', 'Danny Glover'],
                    'Budget': ['37000k', '50000k', '75000k']
                },

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

    dash_table.DataTable(
        id='table-filtered-data',
        columns=[
            {"name": i, "id": i} for i in ['Labels', 'Actual','ModifiedValues']  ##sorted(df.columns)###df.columns
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


@app.callback(Output('table-filtered-data', 'data'),
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
        ###value_store = value_store[0:0]
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
            value_store.at['7', 'Director_1'] = str(selected_landmark)
        if selected_city == 'Actor-1':
            print('Actor-1 Selected')
            Actor_1_value = 'Actor-1='+selected_landmark
            value_store.at['7', 'Actor_1'] = str(selected_landmark)
        if selected_city == 'Actor-2':
            print('Actor-2 Selected')
            Actor_2_value = 'Actor-2='+selected_landmark
            value_store.at['7', 'Actor_2'] = str(selected_landmark)
        if selected_city == 'Budget':
            print('Budget Selected')
            Budget_value = 'Budget='+selected_landmark
            value_store.at['7', 'budget'] = str(selected_landmark)
        ###Final_value = Title_value + Director_value + Actor_1_value + Actor_2_value + Budget_value
    elif n_clicks > 1:
        print('INPUT VALUES-->',selected_country, selected_city, selected_landmark)
        Title_value='Movie-Title = '+selected_country

        print('HMMMMMM------------->',value_store["original_title"].iloc[0])
        value_store.at['7', 'original_title'] = str(selected_country)

        # if value_store["original_title"].iloc[0] != selected_country:
        #      value_store = value_store[0:0]
        #      value_store.at['7', 'original_title'] = str(selected_country)


        if selected_city == 'Director':
            print('Director Selected')
            Director_value = 'Director = ' + selected_landmark
            value_store.at['7', 'Director_1'] = str(selected_landmark)
        if selected_city == 'Actor-1':
            print('Actor-1 Selected')
            Actor_1_value = 'Actor-1='+selected_landmark
            value_store.at['7', 'Actor_1'] = str(selected_landmark)
        if selected_city == 'Actor-2':
            print('Actor-2 Selected')
            Actor_2_value = 'Actor-2='+selected_landmark
            value_store.at['7', 'Actor_2'] = str(selected_landmark)
        if selected_city == 'Budget':
            print('Budget Selected')
            Budget_value = 'Budget='+selected_landmark
            value_store.at['7', 'budget'] = str(selected_landmark)
        ###Final_value = Title_value + Director_value + Actor_1_value + Actor_2_value + Budget_value
    print(value_store.original_title)
    modified_values = 'Movie-Title = '+str(value_store.original_title[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values +'Director = ' + str(value_store.Director_1[0])
    modified_values = modified_values + '-------\n'
    modified_values = modified_values + 'Actor-1 = ' + str(value_store.Actor_1[0])
    modified_values = modified_values +  '-------\n'
    modified_values = modified_values + 'Actor-2 = ' + str(value_store.Actor_2[0])
    modified_values = modified_values +  '-------\n'
    modified_values = modified_values + 'Budget = ' + str(value_store.budget[0])
    print(modified_values)

    whatif_data = value_store.T
    whatif_data = whatif_data.reset_index(drop=False)
    print('Index reset')

    whatif_data.rename(columns={whatif_data.columns[0]: "Labels"}, inplace=True)
    whatif_data.rename(columns={whatif_data.columns[1]: "Values"}, inplace=True)
    print('columns renamed')

    x = df[df['original_title'] == selected_country]
    intro_data = x.T
    intro_data = intro_data.reset_index(drop=False)
    print('Index reset')
    intro_data.rename(columns={intro_data.columns[0]: "Labels"}, inplace=True)
    intro_data.rename(columns={intro_data.columns[1]: "Values"}, inplace=True)
    print('columns renamed')

    merged_left = pd.merge(left=intro_data, right=whatif_data, how='left', left_on='Labels', right_on='Labels')
    merged_left.rename(columns={merged_left.columns[1]: "Actual"}, inplace=True)
    merged_left.rename(columns={merged_left.columns[2]: "ModifiedValues"}, inplace=True)

    print(merged_left.columns)
    print(merged_left)

    merged_left.to_pickle("data/updatedmovielist.pkl")
    whatifdatatable = merged_left.to_dict('records')
    ###value_store =  value_store[0:0]

    return whatifdatatable


@app.callback(
    dash.dependencies.Output('display-predicted-values', 'children'),
    [Input('predict-button-state', 'n_clicks')]
    )
def set_predict_children7(n_clicks):
    if n_clicks > 0:
        return whatif_predict.layout
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)