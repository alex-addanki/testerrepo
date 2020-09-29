from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app


layout = [dcc.Markdown("""
                        ### Model Evaluation Dashboard
                        Movie Content Analysis is intended to access the impact of various influencers on the Movie Revenue or Ratings. 
                        """),
          html.Button(id='explain-button-state', n_clicks=0, children='Explain-IMDB-Score'),

          html.Div(id='explain-content', style={'fontWeight': 'bold'}),
          html.Img(src='Barplot_original_vs_predicted_rand.png')
          ]
@app.callback(Output('explain-content', "children"),
              [Input('explain-button-state', 'n_clicks')])
def explain_output(n_clicks):
    if n_clicks == 1:
        file7 = open('reports\\regression_scores.txt', 'r')
        print('Output of Readlines after writing')
        report = file7.read()
        file7.close()
        print('Read the reports\\regression_scores.txt file and the data is :', report)
        file77 = open('reports\\regression_original_vs_predicted_scores.txt', 'r')
        print("Output of Readlines after writing")
        report = report + '\n'
        report = report + file77.read()
        file77.close()
        print('Read the reports\\regression_original_vs_predicted_scores file and the data is :', report)
    else:
        report=''
        print(report)
    return report
