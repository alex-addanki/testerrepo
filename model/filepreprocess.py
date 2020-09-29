import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import ast
import pickle
style.use('seaborn-poster')
style.use('ggplot')

from bs4 import BeautifulSoup
import requests
import os
import json

def preprocess():
    file = open('data\LX_categ_dataset_unfolded', 'rb')
    all5000_All = pickle.load(file)
    file.close()
    all5000_All = all5000_All.drop(['cast', 'crew', 'Onlycast', 'Director'], axis=1)
    return all5000_All