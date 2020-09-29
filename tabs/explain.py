from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

##atplotlib inline


from app import app


test_png = 'reports\Barplot_original_vs_predicted_rand.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

test_png1 = 'reports\Prediction_Performance_Random.png'
test_base64_1 = base64.b64encode(open(test_png1, 'rb').read()).decode('ascii')

test_png2 = 'reports\Prediction_Performance_Sequence.png'
test_base64_2 = base64.b64encode(open(test_png2, 'rb').read()).decode('ascii')


with open('reports/LimeExplainer.pkl', 'rb') as f:
    exp = pickle.load(f)


cache = "reports\LimeReport.png"
fig = exp.as_pyplot_figure()
fig.set_size_inches(27.5, 10.5)
plt.savefig(cache)



test_base64_3 = base64.b64encode(open(cache, 'rb').read()).decode('ascii')

layout = [dcc.Markdown("""
                        ### Model Prediction Evaluation Dashboard
                        Movie Content Analysis is intended to access the impact of various influencers on the Movie Revenue or Ratings. 
                        """),
          html.Button(id='explain-button-state', n_clicks=0, children='Explain-IMDB-Score'),

          html.Div(id='explain-content', style={'fontWeight': 'bold'}),
          html.Img(src='data:image/png;base64,{}'.format(test_base64_3), style={'height':'100%', 'width':'100%'}),
          ###html.Img(src='data:image/png;base64,{}'.format(test_base64)),
          ###html.Img(src='data:image/png;base64,{}'.format(test_base64_1)),
          ###html.Img(src='data:image/png;base64,{}'.format(test_base64_2)),

          ###html.Img(src=fig),
          ###html.Img(src='data:image/png;base64,{}'.format(fig)),
          ###html.Div([dcc.Graph(figure=fig)])
          ]
@app.callback(Output('explain-content', "children"),
              [Input('explain-button-state', 'n_clicks')])
def explain_output(n_clicks):
    if n_clicks == 1:
        report = "\t"
        file7 = open('reports\\regression_scores.txt', 'r')
        print('Output of Readlines after writing')
        for i in file7.read():
            if '\n' in i:
                report += "\n \t"
            else:
                report += i
        file7.close()

        report += "\n \t"
        report += "\n \t"


        print('Read the reports\\regression_scores.txt file and the data is :', report)
        file77 = open('reports\\regression_original_vs_predicted_scores.txt', 'r')
        print("Output of Readlines after writing")
        ###report = report + '\n'
        ###report = report + file77.read()
        for i in file77.read():
            if '\n' in i:
                report += "\n \t"
            else:
                report += i
        file77.close()

        report += "\n \t"
        report += "\n \t"

        print('Read the reports\\regression_original_vs_predicted_scores file and the data is :', report)
    else:
        report=''
        print(report)
    return report
