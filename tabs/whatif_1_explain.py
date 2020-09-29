import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

import flask
import glob
import os

image_directory = 'reports'
###list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
list_of_images = ['LimeReport.png','VAG_LIME_EXPLAINER.png']

print(list_of_images)
static_image_route = '/static/'


from app import app


layout = html.Div([
    dcc.Dropdown(
        id='image-dropdown',
        options=[{'label': i, 'value': i} for i in list_of_images],
        value=list_of_images[0]
    ),
    html.Img(id='image')
])

@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(value):
    print('------------###',value)
    return static_image_route + value

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    print(os.path.abspath(os.getcwd()))
    print('---',image_path)
    image_name = '{}.png'.format(image_path)
    print('---', image_name)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)