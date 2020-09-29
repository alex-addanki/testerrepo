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

def preprocess_demo():
    file = open('data\DEMO_ALL_Features_AUG25', 'rb')
    all5000_All = pickle.load(file)
    file.close()
    all5000_All = all5000_All.drop(['id','cast', 'crew','Director_1', 'Director_2', 'Director_3', 'Actor_1', 'Actor_2','Actor_3', 'Actor_4', 'Actor_5'], axis=1)
    return all5000_All