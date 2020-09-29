from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

from app import app



img = cv2.imread("reports\VAG_LIME_EXPLAINER.png")
alpha = 2.0
beta = -160
new = alpha * img + beta
new = np.clip(new, 0, 255).astype(np.uint8)
cv2.imwrite("reports\VAG_LIME_EXPLAINER_1.png", new)
whatifexplain_lime_png7 = 'reports\VAG_LIME_EXPLAINER_1.png'

###image_lime = Image.open('reports\VAG_LIME_EXPLAINER.png')
###new_image_lime = image_lime.resize((25, 25))
###new_image_lime.save(whatifexplain_lime_png7)
whatifexplain_lime_png_base64 = base64.b64encode(open(whatifexplain_lime_png7, 'rb').read()).decode('ascii')

whatifexplain_speedo_png = 'images\Modifiedpredicted.png'
whatifexplain_speedo_png_base64 = base64.b64encode(open(whatifexplain_speedo_png, 'rb').read()).decode('ascii')


layout = [dcc.Markdown("""
                        ### Model WHAT-IF Prediction Evaluation Dashboard
                        Movie Content Analysis is intended to access the impact of various influencers on the Movie Revenue or Ratings. 
                        """),
          html.Img(src='data:image/png;base64,{}'.format(whatifexplain_speedo_png_base64)),
          html.Hr(),
          html.Img(src='data:image/png;base64,{}'.format(whatifexplain_lime_png_base64))  ###, style={'height':'100%', 'width':'100%'}
          ]

