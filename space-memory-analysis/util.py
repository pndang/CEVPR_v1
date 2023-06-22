import os
import pandas as pd
import numpy as np


# Function for calculating rolling averages using queue data structure
def calculate_rolling_avg(dataframe, column_idx, num_rows):

  """
  Calculate rolling averages of the data field at 'column_idx' by averaging
  'num_rows' rows prior and after.

  Params:
  dataframe (DataFrame): subject dataframe containing the column to calculate
    rolling averages
  column_idx (int): index of the column in {dataframe}
  num_rows (int): number of rows before and after an entry to calculate
    rolling average for that entry
  
  Return:
  None

  """

  averages = []
  q = []
  total_rows = dataframe.shape[0]
  data = list(dataframe.iloc[:, column_idx])
  max_size = num_rows*2+1
  pointer = 0
  pointer_offset = 0

  for i in range(total_rows):

    # For the first {num_rows} entries, calculate average using all entries
    # before and {num_rows} entries after an entry
    if i < num_rows:
      q.append(data[i])
      values = data[:i] + data[i: i+num_rows+1]

    # If queue length is less than max size, and 
    elif len(q) != max_size and i < max_size:
      q.append(data[i])
      pointer_offset += 1
      if len(q) == max_size:
        pointer = i - pointer_offset
      continue
    elif i == total_rows-1:
      averages.append(np.mean(q))
      del q[0]
      q.append(data[i])
      for j in range(pointer_offset):
        values = data[pointer-num_rows:]
        averages.append(np.mean(values))
      continue
    else:
      values = q.copy()
      del q[0]
      q.append(data[i])
      pointer += 1
    averages.append(np.mean(values))      

  dataframe['rolling_avg'] = averages
  
  return None
