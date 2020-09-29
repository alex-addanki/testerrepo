from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

import PIL
from PIL import Image


from app import app


avatar_png = 'images\Tangled.jpg'
avatar_base64 = base64.b64encode(open(avatar_png, 'rb').read()).decode('ascii')

avatar_speedo_png = 'images/predicted.png'
image = Image.open(avatar_speedo_png)

avatar_speedo_base64 = base64.b64encode(open(avatar_speedo_png, 'rb').read()).decode('ascii')


layout = html.Div([
          html.Img(src='data:image/png;base64,{}'.format(avatar_base64)), html.Img(src='data:image/png;base64,{}'.format(avatar_speedo_base64)),
], style={'textAlign': 'left'})
