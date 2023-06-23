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

  Notes:
    - The working data point for which its rolling average is calculated is 
      referred to as "main"
    - pointer is always the middle element of the queue, except when calculating
      averages for the last {num_rows} entries
    - pointer_offset is always equal to num_rows, pointer_offset keeps track of 
      how much the iterating index i is ahead of pointer by. Because of this
      offset, we use a loop that runs pointer_offset times to calculate rolling
      averages for the remaining entries in the 2nd elif 

  """

  averages = []
  q = []
  total_rows = dataframe.shape[0]
  data = list(dataframe.iloc[:, column_idx])
  max_size = num_rows*2+1
  pointer = 0
  pointer_offset = 0

  for i in range(total_rows):

    # For the first {num_rows+1} entries, calculate average using all entries
    # before and {num_rows} entries after main
    if i <= num_rows:
      q.append(data[i])
      values = data[:i] + data[i: i+num_rows+1]

    # If index of main is less than max size (queue is not full yet), add main
    # to queue and increase pointer_offset by one. This is to fill up the queue,
    # once queue is full, set pointer to the middle of queue by subtracting
    # current index by pointer_offset. No averages is added in this step.
    elif i < max_size:
      q.append(data[i])
      pointer_offset += 1
      if len(q) == max_size:
        pointer = i - pointer_offset
      continue

    # In the last iteration of the loop, calculate the last {num_rows} 
    # rolling averages using a loop that runs pointer_offset times, note that i 
    # is always off from pointer by pointer_offset. 
    elif i == total_rows-1:
      averages.append(np.mean(q))
      del q[0]
      q.append(data[i])

      # For the last {num_rows} entries, calculate average using {num_rows}
      # entries before and all entries after main
      for j in range(pointer_offset):
        values = data[pointer-num_rows:]
        averages.append(np.mean(values))
        pointer += 1
      continue

    # When queue is nice and full, and we're not calculating average for the 
    # first, or last, {num_rows} entries (edge cases); calculate average of the 
    # queue, remove first queue element, add next data entry to queue, and move
    # pointer by one
    else:
      values = q.copy()
      del q[0]
      q.append(data[i])
      pointer += 1
    averages.append(np.mean(values))      

  dataframe['rolling_avg'] = averages
  
  return None
