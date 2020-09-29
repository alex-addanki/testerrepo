from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from app import app, server
from tabs import introduction,intro, whatif_demo

style = {'maxWidth': '960px', 'margin': 'auto'}

from model import filepreprocess

tabs_styles = {
    'height': '44px',
    'width': "calc(100% - 100px)"
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    dcc.Markdown('# Content Analysis', style=tab_style),
    dcc.Tabs(id='tabs', value='introduction', children=[
                                                    dcc.Tab(label='Model Evaluation'             , value='introduction', style=tab_style, selected_style=tab_selected_style),
                                                    dcc.Tab(label='Predictions'             , value='tab-intro', style=tab_style, selected_style=tab_selected_style),
                                                    dcc.Tab(label='Whatif'             , value='whatif-demo', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='Model Evolution'             , value='tab-viewall', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='Model Predictions'          , value='tab-pred', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='Model Explainability'       , value='tab-explain', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='What if Scenario'           , value='tab-whatif', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='What if Scenario Explain'           , value='tab-exlain-whatif', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='What if Scenario-2'           , value='tab-whatif2', style=tab_style, selected_style=tab_selected_style),
                                                    ###dcc.Tab(label='Evaluate the Content'       , value='tab-evaluate', style=tab_style, selected_style=tab_selected_style),
                                                ]),
    html.Div(id='tabs-content'),
], style=tabs_styles)  ###style  tabs_styles

@app.callback(Output('tabs-content', 'children'),
             [Input('tabs', 'value')])
def render_content(tab):
    if   tab == 'introduction' : return introduction.layout
    elif tab   == 'tab-intro'   : return intro.layout
    elif tab   == 'whatif-demo' : return whatif_demo.layout
    ###elif tab   == 'tab-pred'    : return predict.layout
    ###elif tab   == 'tab-explain' : return explain.layout
    ###elif tab   == 'tab-whatif'  : return whatif.layout
    ###elif tab == 'tab-exlain-whatif'  : return whatif_1_explain.layout
    ###elif tab == 'tab-whatif2'    : return whatif_2.layout
    ###elif tab   == 'tab-evaluate': return evaluate.layout

if __name__ == '__main__':
    app.run_server(debug=True)
