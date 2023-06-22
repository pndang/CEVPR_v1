import datetime
import pandas as pd 
import numpy as np
import plotly.express as px
import seaborn as sns
import os 
import pickle
import sys
from util import *
import gzip
import time

import warnings
warnings.filterwarnings('ignore')

path = 'C:/Users/phuro/UCSD/NiemaLab/data/'
fp = os.path.join(path, 'covid19variants.csv')
df = pd.read_csv(fp)

# Creating a dictionary of variants (keys are variant names & values are 
# subset dataframes of variants)
variants = df['variant_name'].unique()
variants_dict = {var: df[df['variant_name'] == var] for var in variants}

# Calculating rolling averages for each variant dataframes in dictionary 
for key, val in variants_dict.items():
    val.reset_index(inplace=True)
    calculate_rolling_avg(val, 5, 3)
    val.set_index('index', inplace=True)

# Concatenating the variant-specific dataframes into one 
data = pd.concat(variants_dict.values())
data = data.dropna(subset=['date'])

# Export to data folder
wp = os.path.join(path, 'covid19-variants-w-rolling-avg.csv')
data.to_csv(wp)

results = open('results.txt', 'w')

# Get memory size of resulting dataset object/file
object_size = sys.getsizeof(data)
result = f'Memory size of dataset object: {object_size} bytes'
print(result); results.write(result + '\n')

file_size = os.stat(wp).st_size
result = f'Memory size of dataset file: {file_size} bytes\n'
print(result); results.write(result + '\n')

# Pickle dataset
wp = os.path.join(path, 'covid19-variants-w-rolling-avg.pickle')
with open(wp, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Get memory size of resulting dataset pickle file
start_time = time.time()

with open(wp, 'rb') as f:
    df_from_pickle = pickle.load(f)

end_time = time.time()
execution_time = end_time - start_time

obj_size_pkl = sys.getsizeof(df_from_pickle)
result = f'Memory size of dataset object (pickled): {obj_size_pkl} bytes'
print(result); results.write(result + '\n')

file_size_pkl = os.stat(wp).st_size
result = f'Memory size of pickle file: {file_size_pkl} bytes'
print(result); results.write(result + '\n')
decomp_time = f'Decompression time: {execution_time} seconds\n'
print(decomp_time); results.write(decomp_time + '\n')

# Compress pickle file with gzip
wp = os.path.join(path, 'covid19-variants-w-rolling-avg.pkl.gz')
with gzip.open(wp, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Get memory size of compressed pickle file
start_time = time.time()

with gzip.open(wp, 'rb') as f:
    start_time = time.time()
    df_pkl_zipped = pickle.load(f)

end_time = time.time()
execution_time = end_time - start_time

obj_size_gzipped = sys.getsizeof(df_pkl_zipped)
result = f'Memory size of dataset object (pickled & zipped): {obj_size_gzipped} bytes'
print(result); results.write(result + '\n')

file_size_gzipped = os.stat(wp).st_size
result = f'Memory size of gzip file: {file_size_gzipped} bytes'
print(result); results.write(result + '\n')

decomp_time = f'Decompression time: {execution_time} seconds\n'
print(decomp_time); results.write(decomp_time + '\n')

results.close()
