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

avatar_png = 'reports\imdm_actual.png'
avatar_base64 = base64.b64encode(open(avatar_png, 'rb').read()).decode('ascii')

avatar_png = 'reports\imdm_actual.png'
avatar_base642 = base64.b64encode(open(avatar_png, 'rb').read()).decode('ascii')


##htm = 'model\oi.html'
###htmimg = base64.b64encode(open(htm, 'rb').read()).decode('ascii')



layout = html.Div([
          html.Img(src='data:image/png;base64,{}'.format(avatar_base64)),
          html.Img(src='data:image/png;base64,{}'.format(avatar_base642)),
          html.A('Navigate to LIME-Explainer', href='file:///C:/LX_DOCS/LX_Learning/Kaggle/Movie_Content_Prediction/Dash/model/oi.html', target='_blank'),
], style={'textAlign': 'left'})

