import os
import glob
import pandas as pd

def generate_explain_df():
    ###os.chdir("C:\\LX_DOCS\\LX_Learning\\Kaggle\\Movie_Content_Prediction\\Dash\\reports")
    extension = 'txt'
    all_reports = [i for i in glob.glob('reports\\*.{}'.format(extension))]
    df777 = pd.concat([pd.read_csv(f,delimiter=':') for f in all_reports ])
    print(df777)
    return df777