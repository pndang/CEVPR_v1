import datetime
import pandas as pd 
import numpy as np
import plotly.express as px
import seaborn as sns
import os 
import pickle

import warnings
warnings.filterwarnings('ignore')

path = 'C:/Users/phuro/UCSD/NiemaLab/data/'
fp = os.path.join(path, 'covid19variants.csv')
df = pd.read_csv(fp)

