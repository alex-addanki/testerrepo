import dash
import dash_bootstrap_components as dbc

###external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
###external_stylesheets = ["https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])   ###external_stylesheets  ###[dbc.themes.CYBORG]

###COSMO,FLATLY,LUMEN,MINTY,SIMPLEX,JOURNAL,LUX ,, SKETCHY ,,,

card_style = {
    "box-shadow": "0 4px 5px 0 rgba(0,0,0,0.14), 0 1px 10px 0 rgba(0,0,0,0.12), 0 2px 4px -1px rgba(0,0,0,0.3)"
}

app.config.suppress_callback_exceptions = True
server = app.server
