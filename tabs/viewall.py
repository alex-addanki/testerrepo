import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from joblib import load
import numpy as np



from app import app
from model import filepreprocess

##df = pd.read_csv('data\dataset_model.csv')
df= filepreprocess.preprocess()

PAGE_SIZE = 20

layout = html.Div([
    dcc.Input(value='', id='filter-input', placeholder='Filter', debounce=True),
    dash_table.DataTable(
        id='datatable-paging',
        columns=[
            {"name": i, "id": i} for i in df.columns  # sorted(df.columns)
        ],
        page_current=0,
        page_size=PAGE_SIZE,
        page_action='custom',

        sort_action='custom',
        sort_mode='single',
        sort_by=[]
    )
])


@app.callback(
    Output('datatable-paging', 'data'),
    [Input('datatable-paging', 'page_current'),
     Input('datatable-paging', 'page_size'),
     Input('datatable-paging', 'sort_by'),
     Input('filter-input', 'value')])
def update_table(page_current, page_size, sort_by, filter_string):
    # Filter
    ###dff = df[df.apply(lambda row: row.str.contains(filter_string, regex=False,case=False).any(), axis=1)]
    dff = df[df['original_title'].str.contains(filter_string, regex=False,case=False)]
    # Sort if necessary
    if len(sort_by):
        dff = dff.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )

    return dff.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)