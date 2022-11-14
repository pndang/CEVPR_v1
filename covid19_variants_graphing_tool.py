# Author: Phu Dang
# Date: 11.8.2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime
sns.set_theme(style="whitegrid")
import random

import warnings
warnings.filterwarnings('ignore')


# all possible value for color palettes
all_palettes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',          
                'GnBu', 'GnBu_r',  'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',  'PRGn', 'PRGn_r', 'Paired', 'Paired_r',                 
                'Pastel1', 'Pastel1_r', 'Pastel2',  'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',                 
                'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1',                 
                'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr',                 
                'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r',  'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',                
                'cividis', 'cividis_r', 'cool', 'cool_r',  'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',  'gist_earth',                
                'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',  'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',                
                'gist_stern_r', 'gist_yarg',  'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r',  'hsv', 'hsv_r', 'icefire',               
                'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r',  'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink',                
                'pink_r',  'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',  'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',               
                'summer_r', 'tab10', 'tab10_r','tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo',  'turbo_r', 'twilight',                
                'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',  'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']


# Function for calculating rolling averages
def calculate_rolling_avg(dataframe):
  dataframe['rolling_avg'] = 0
  for i in range(0, dataframe.shape[0]):
    idx = i
    last_idx = dataframe.shape[0]-1
    count = 0  # count number of values added (imaginary), to handle the first and last 3 rows
               # Idea: if index is 0, assume 3 "before" values already added, 
               #       if index is 1, assume 2 "before" values already added, 
               #       so on

    values_before = []
    while len(values_before) < 3 and count < 3:
      if i == 0:
        break
      count = 2 if idx == 1 else 1 if idx == 2 else 0
      values_before.append(dataframe.iloc[idx-1][5])
      idx -= 1
      count += 1   
    
    idx, count = i, 0
    values_after = []
    while len(values_after) < 3 and count < 3:
      if i == last_idx:
        break
      count = 2 if idx == last_idx-1 else 1 if i == last_idx-2 else 0
      values_after.append(dataframe.iloc[idx+1][5])
      idx += 1
      count += 1

    # Calculate the average
    average = np.mean(values_before + [dataframe.iloc[i][5]] + values_after)
    dataframe.at[i, 'rolling_avg'] = average

    # print(i)
    # print(values_before)
    # print(values_after)
    # print(values_before + [dataframe.iloc[i][4]] + values_after)
    # print(average)
    # print('***********')

  return None


# Importing dataset
data = pd.read_csv("covid19variants.csv")

# ******************************************************************************

# Function to display user input choices in command prompt
def user_pick(options, option_type):
    print("\nPlease choose "+str(option_type)+":")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1, element))
    choice = int(input("\nEnter number {}-{}: ".format(1, len(options))))
    if choice < 1 or choice > len(options):
        print("\nInvalid input, please enter one of the choices below: ")
        return user_pick(options, option_type)
    return options[int(choice)-1]

years = [2021]
months = {'January':1,
          'February':2, 
          'March':3, 
          'April':4, 
          'May':5, 
          'June':6, 
          'July':7, 
          'August':8, 
          'September':9, 
          'October':10, 
          'November':11, 
          'December':12}

# Choosing starting date in command prompt
print("\nPlease enter your starting period of analysis:")
start_yr = user_pick(years, 'starting year')
start_mo = user_pick(list(months.keys()), 'starting month')
start_mo = months[start_mo]

# Choosing ending date in command prompt
print("\nPlease enter your ending period of analysis:")

end_yr = user_pick(years, 'ending year')
if end_yr < start_yr:
    print("\nInvalid input, please enter an ending year after starting year")
    valid = False
    while not valid:
        end_yr = user_pick(years, 'ending year')
        if end_yr >= start_yr:
            valid = True

if end_yr == start_yr:
    month_lst = list(months.keys())
    end_mo = user_pick(month_lst[month_lst.index([k for k, v in months.items() if v == start_mo][0])+1:], 'ending month')
else:
    end_mo = user_pick(list(months.keys()), 'ending month')
end_mo = months[end_mo]

print("Generating graph... ")

start = datetime.datetime(start_yr, start_mo, 1) # starting date
end = datetime.datetime(end_yr, end_mo, 30 if end_mo in [4,6,9,11] else 28 if end_mo in [2] else 31) # ending date
variant_col = 'variant_name' # change for different datasets
date_col = 'date' # change for different datasets


# ******************************************************************************


# Check if time_col column is of type datetime, proceed if yes, convert to datetime if no
if data.get(date_col).dtypes == '<M8[ns]':
  pass
else:
  data['datetime'] = pd.to_datetime(data[date_col])
  date_col = 'datetime'

data = data[(data[date_col] >= start) & (data[date_col] <= end)]

variants = data.get(variant_col).unique()

# Creating a dictionary of variants (keys are variant names & values are subset dataframes of variants)
variants_dict = {var: data[data[variant_col] == var] for var in variants}

# Calculating rolling averages for each variant dataframes in dictionary 
for value in variants_dict.values():
  value.reset_index(inplace=True)
  calculate_rolling_avg(value) # calculate_rolling_avg function is defined above 
  value.set_index('index', inplace=True)

# Concatenating the dataframes into one for plotting, drop rows with N/A
variants_wra = pd.concat(variants_dict.values())
variants_wra = variants_wra.dropna(subset=[date_col])

# Plotting the rolling averages of daily specimens count by variant
plt.figure(figsize=(15, 9))

# Randomly choose a color palette
color_palette = random.choice(all_palettes)

# Displaying info about plot
print("Start: "+str(start))
print("End: "+str(end))
print("Color palette: "+color_palette)
print()

fig = sns.lineplot(
    data=variants_wra, 
    x=date_col, 
    y='rolling_avg', 
    hue=variant_col,
    palette=color_palette
    )

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title("Specimen Count by Variant (averaged 3 days prior/after)")
plt.show()


# Unused Code Archive:

    # if end_yr == start_yr and end_mo <= start_mo:
    #     print("Invalid input, please enter an ending month after starting month")
    #     valid = False
    #     while not valid:
    #         end_mo = user_pick(months, 'ending month')
    #         if end_mo > start_mo:
    #             valid = True


# Action items (week 7)
 # Include days for users, pywidgets to select certain variants, can we use a calendar for user inputs?
 # Long term goal: a script that a user picks on and a gouie will show up for user inputs
 # Add start month to end months if same year
 # CREATE AN APPLET !! Using Plotly
