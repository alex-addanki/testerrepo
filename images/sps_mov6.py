import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from app import app
import flask
import glob
import os

image_directory = 'images'
list_of_images = ['predicted.png']
avatar_png = 'images\Avengers_ Age of Ultron.jpg'
avatar_base64 = base64.b64encode(open(avatar_png, 'rb').read()).decode('ascii')
static_image_route6 = '/static/'


layout = html.Div([
    html.Button(id='show-whatifbutton-speedo6', n_clicks=0, children=''),
    html.Img(src='data:image/png;base64,{}'.format(avatar_base64)),
    html.Img(id='image7776')
])

@app.callback(
    dash.dependencies.Output('image7776', 'src'),
    [dash.dependencies.Input('show-whatifbutton-speedo6', 'value')])
def update_image_src6(value):
    value = list_of_images[0]
    return static_image_route6 + value

@app.server.route('{}<image_path>.png'.format(static_image_route6))
def serve_image6(image_path):
    print(os.path.abspath(os.getcwd()))
    print('---',image_path)
    image_name = '{}.png'.format(image_path)
    print('---', image_name)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)