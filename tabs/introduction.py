from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64

from app import app

img1_1 = 'images\Data_Distribution_by_Generes.png'
img1 = base64.b64encode(open(img1_1, 'rb').read()).decode('ascii')

img2_1 = 'images\stacked_model7_July15_2020-Copy1.png'
img2 = base64.b64encode(open(img2_1, 'rb').read()).decode('ascii')

poster_1 = 'images\Moviecollage3.png'
img3 = base64.b64encode(open(poster_1, 'rb').read()).decode('ascii')

plot_1 = 'images\plotScrape.png'
img4 = base64.b64encode(open(plot_1, 'rb').read()).decode('ascii')

model_1 = 'images\Model_Architecture.png'
img5 = base64.b64encode(open(model_1, 'rb').read()).decode('ascii')


train_1 = 'images\Training_Accuracy.png'
img6 = base64.b64encode(open(train_1, 'rb').read()).decode('ascii')

train_2 = 'images\Training_MAE.png'
img7 = base64.b64encode(open(train_2, 'rb').read()).decode('ascii')

train_3 = 'images\Training_MSE.png'
img8 = base64.b64encode(open(train_3, 'rb').read()).decode('ascii')

train_4 = 'images\Validation_Metrics.png'
img9 = base64.b64encode(open(train_4, 'rb').read()).decode('ascii')

train_5 = 'images\ModelPerformanceMetrics.png'
img10 = base64.b64encode(open(train_5, 'rb').read()).decode('ascii')


Obs_1 = 'images\TextObservation.png'
img11 = base64.b64encode(open(Obs_1, 'rb').read()).decode('ascii')

gc_1 = 'images\GRADCAM_INTERPRETY\mov_1_1.png'
img12 = base64.b64encode(open(gc_1, 'rb').read()).decode('ascii')

gc_1_1 = 'images\GRADCAM_INTERPRETY\mov_1_2.png'
img12_1 = base64.b64encode(open(gc_1_1, 'rb').read()).decode('ascii')

gc_2 = 'images\GRADCAM_INTERPRETY\mov_2_1.png'
img13 = base64.b64encode(open(gc_2, 'rb').read()).decode('ascii')

gc_2_1 = 'images\GRADCAM_INTERPRETY\mov_2_2.png'
img13_1 = base64.b64encode(open(gc_2_1, 'rb').read()).decode('ascii')

gc_2_1 = 'images\GRADCAM_INTERPRETY\mov_2_2.png'
img13_1 = base64.b64encode(open(gc_2_1, 'rb').read()).decode('ascii')

gc_2_3 = 'images\Categorica_Results.png'
img14 = base64.b64encode(open(gc_2_3, 'rb').read()).decode('ascii')

gc_2_4 = 'images\Categorica_Results2.png'
img15 = base64.b64encode(open(gc_2_4, 'rb').read()).decode('ascii')





colors = {
    'background': '#111111',
    'text': '#7FFBFF'
}

layout = html.Div([
    html.Br(),html.Br(),
    html.H1(children='About the Data',style={'textAlign': 'left'}),
        html.Br(),html.Br(),html.Br(),
        dcc.Markdown("""### Data Distribution by the Genres and the Number of Movies we have scraped """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img1)),

         html.Br(),html.Br(),html.Br(),
         dcc.Markdown("""### A Few Scraped posters """),
         html.Br(),html.Br(),html.Br(),
         html.Img(src='data:image/png;base64,{}'.format(img3)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### A Few Scraped Plots """),
        html.Br(), html.Br(), html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img4)),

    html.Br(), html.Br(),html.Br(),
    html.H1(children='About the Model', style={'textAlign': 'left'}),
        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Design """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img5)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Architecture """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img2)),

    html.Br(), html.Br(), html.Br(),
    html.H1(children='Model Training\Validation Metrics', style={'textAlign': 'left'}),
        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Training Accuracy """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img6)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Training Mean Absolute Error """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img7)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Training Mean Squared Error """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img8)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Validation Performance """),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img9)),

        html.Br(), html.Br(), html.Br(),
        dcc.Markdown("""### Model Validation Performance Metrics"""),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(img10)),

    html.Br(), html.Br(), html.Br(),
    html.H1(children='Model Results', style={'textAlign': 'left'}),
    html.Br(), html.Br(), html.Br(),
    dcc.Markdown("""### Model Textual Observations """),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img11)),
    html.Br(),
    dcc.Markdown("""### Model Poster Observations """),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img12)),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img12_1)),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img13)),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img13_1)),
    html.Br(),
    dcc.Markdown("""### Model Cast Director Observations """),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img14)),
    html.Br(),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(img15)),
    html.Br(),

], style={'textAlign': 'center'})
