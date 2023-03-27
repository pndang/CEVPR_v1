import datetime
from dash import Dash, dcc, html, ALL, ctx
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd 
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import seaborn as sns
from PIL import Image, ImageDraw
import os 
import warnings
warnings.filterwarnings('ignore')
# import cProfile
# import re
# cProfile.run('re.compile("foo|bar")')
# from werkzeug.middleware.profiler import ProfilerMiddleware
from timeit import default_timer as timer
import dash
# from queue import Queue

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server 


app.config.suppress_callback_exceptions = True

# ------------------------------------------------------------------------------

# Function for calculating rolling averages
# def calculate_rolling_avg(dataframe, column_idx, num_rows):

#   """Calculate rolling averages of the data field at 'column_idx' by averaging
#   'num_rows' rows prior and after."""

#   dataframe['rolling_avg'] = 0
#   for i in range(0, dataframe.shape[0]):
#     idx = i
#     last_idx = dataframe.shape[0]-1
#     count = 0  # count number of values added (imaginary), to handle the first and last num_rows rows
#                # Idea: if index is 0, assume num_rows "before" values already added, 
#                #       if index is 1, assume num_rows-1 "before" values already added, 
#                #       if index is -3, assume num_rows-2 "after" values already added,
#                #       so on
#     values_before = []
#     while len(values_before) < num_rows and count < num_rows:
#       if i == 0:
#         break
#       elif i < num_rows:
#         break
#       values_before.append(dataframe.iloc[idx-1][column_idx])
#       idx -= 1
#       count += 1   
    
#     idx, count = i, 0
#     values_after = []
#     while len(values_after) < num_rows and count < num_rows:
#       if i == last_idx:
#         break
#       elif dataframe.shape[0]-i <= num_rows:
#         values_after = list(dataframe[dataframe.columns[column_idx]][i+1:])
#         break
#       values_after.append(dataframe.iloc[idx+1][column_idx])
#       idx += 1
#       count += 1

#     # Calculating the average
#     average = np.mean(values_before + [dataframe.iloc[i][column_idx]] + values_after)
#     dataframe.at[i, 'rolling_avg'] = average

#   return None

# Function for calculating rolling averages using queue data structure
def calculate_rolling_avg(dataframe, column_idx, num_rows):

  """Calculate rolling averages of the data field at 'column_idx' by averaging
  'num_rows' rows prior and after."""

  averages = []
  q = []
  total_rows = dataframe.shape[0]
  data = list(dataframe.iloc[:, column_idx])
  max_size = num_rows*2+1
  pointer = 0
  pointer_offset = 0

  for i in range(total_rows):
    if i < num_rows:
      q.append(data[i])
      values = data[:i] + data[i: i+num_rows+1]
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

  # print(averages)
  # print(f'length: {len(averages)}')
  dataframe['rolling_avg'] = averages
  
  return None

# Importing and cleaning dataset
if 'df' not in locals():
    df = pd.read_csv('covid19variants.csv')

# ------------------------------------------------------------------------------

# WebApp layout with BOOTSTRAP theme stylesheet for component positioning
# Check 'covid19_variants_graphing_tool.py' for regular layout

# ------------------------------------------------------------------------------

app.layout = dbc.Container([
  dbc.Row([
    dbc.Col([
      html.Br(),
      html.H4("COVID-19 Variants Graphing App", style={'textAlign':'center'}),
      html.Br()
      ], width=12)
  ]),

  dbc.Row([
    dbc.Col([
      dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=datetime.date(2021, 1, 1),
        max_date_allowed=datetime.date(2022, 9, 23),
        initial_visible_month=datetime.date(2021, 1, 1),
        end_date=datetime.date(2022, 9, 23)
    ), 
    html.Br(), 
    html.Br()], width=6),

    dbc.Col([
      dcc.Dropdown(
        id='mydropdown',
        options={x: x for x in df.variant_name.unique()},
        value='all',
        multi=True,
        placeholder='Select variants'
      )], width=6)
  ]),

  dbc.Row([
    dbc.Col([
      html.Div(id='output-date-picker-range'), 
      html.Br()
    ], width=12),

    html.Br(),

  ]),

  dbc.Row([
    dbc.Col([
        dbc.Button(
        "Choose color palette",
        id="mypalette",
        className="me-1"
    ),

    dbc.Popover(
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    [html.Img(src="/assets/inferno.png", height=30), "inferno"],
                    id={"type": "list-group-item", "index": "inferno"}
                ),
                dbc.ListGroupItem(
                    [html.Img(src="/assets/icefire.png", height=30), "icefire"],
                    id={"type": "list-group-item", "index": "icefire"}
                ),
                dbc.ListGroupItem(
                    [html.Img(src="/assets/rainbow.png", height=30), "rainbow"],
                    id={"type": "list-group-item", "index": "rainbow"}
                ),
                dbc.ListGroupItem(
                    [html.Img(src="/assets/autumn.png", height=30), "autumn"],
                    id={"type": "list-group-item", "index": "autumn"}
                ),
                dbc.ListGroupItem(
                    [html.Img(src="/assets/ocean.png", height=30), "ocean"],
                    id={"type": "list-group-item", "index": "ocean"}
                ),
            ]
        ),
        target="mypalette",
        body=True,
        trigger="hover",
    )
    ], width=6),

    dbc.Col([
        html.Div(
        dcc.RadioItems(options=[
        {'label': 'raw data', 'value': 0},
        {'label': 'smoothed', 'value': 3}],
        id='smoothing-period', value=3, inline=True,
        labelStyle={'padding-left':42})
    )], width=6)
  ]),

  html.Br(),

  dcc.Graph(id='my_plot')
  
])


@app.callback(
    [Output('output-date-picker-range', 'children'),
    Output('my_plot', 'figure')],
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('mydropdown', 'value'),
    Input({'type': 'list-group-item', 'index': ALL}, 'n_clicks'),
    Input('smoothing-period', 'value'),
    prevent_initial_call=True)


def update_output(start_date, end_date, dropdown_val, palette, smoothing_period):

    if ctx.triggered_id.index not in ['inferno', 'icefire', 'rainbow', 'autumn', 'ocean']:
        palette = 'turbo'
    else: 
        palette = ctx.triggered_id.index

    string_prefix = 'You have selected: '
    plot_ready = False
    if start_date is not None:
        start_date_obj = datetime.date.fromisoformat(start_date)
        start_date_str = start_date_obj.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: {} | '.format(start_date_str)
        plot_ready = True
    if end_date is not None:
        end_date_obj = datetime.date.fromisoformat(end_date)
        end_date_str = end_date_obj.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: {}'.format(end_date_str)
    if smoothing_period is not None:
        string_prefix = string_prefix + ' | Smoothing range: {} days'.format(smoothing_period*2)
    if plot_ready and dropdown_val != None:
        if dropdown_val == 'all' or dropdown_val == []:
            data = df.copy()
        else:
            data = df.loc[df['variant_name'].isin(dropdown_val)]
        date_col = 'date'
        variant_col = 'variant_name'

        # Check if time_col column is of type datetime, proceed if yes, convert 
        # to datetime if no
        if data.get(date_col).dtypes == '<M8[ns]':
            pass
        else:
            data['date'] = [datetime.datetime.strptime(date, '%Y-%m-%d').date()
                            for date in data[date_col]]
            date_col = 'date'

        data = data[(data[date_col] >= start_date_obj) & \
                    (data[date_col] <= end_date_obj)]

        variants = data.get(variant_col).unique()

        palette_reformatted = sns.color_palette(palette, len(variants)).as_hex()

        if smoothing_period == 0:
            fig = px.line(data_frame=data[data[variant_col].isin(variants)],
                      x=date_col,
                      y='specimens',
                      color=variant_col,
                      color_discrete_sequence=palette_reformatted)

            fig.update_layout(title='Daily Specimen Count by Variant',
                              yaxis_title='specimen count')
            
            return f'{string_prefix} | Palette: {palette}', fig

        # Creating a dictionary of variants (keys are variant names & values are 
        # subset dataframes of variants)
        variants_dict = {var: data[data[variant_col] == var] for var in variants}

        # Profiling
        n = 50
        times = []
        for ite in range(n):
          task_name = f"task_{ite+1}"
          start_1 = timer()
          # Calculating rolling averages for each variant dataframes in dictionary 
          for key, value in variants_dict.items():
              value.reset_index(inplace=True)
              calculate_rolling_avg(value, 5, smoothing_period)
              value.set_index('index', inplace=True)
          # dash.callback_context.record_timing(task_name, timer() - start_1, '1st task')
          time = timer() - start_1
          times.append(time)
          print(f"Iteration {ite+1}/{n}: {time}")                
        print(np.mean(times))

        start_2 = timer()
        # Concatenating the dataframes into one for plotting, drop rows with N/A
        variants_wra = pd.concat(variants_dict.values())
        variants_wra = variants_wra.dropna(subset=[date_col])
        dash.callback_context.record_timing('task_2', timer() - start_2, '2nd task')

        fig = px.line(data_frame=variants_wra,
                      x=date_col,
                      y='rolling_avg',
                      color=variant_col,
                      color_discrete_sequence=palette_reformatted)

        fig.update_layout(title=f'Daily Specimen Count by Variant, averaged {smoothing_period} days prior/after',
                          yaxis_title='specimen count')

        return f'{string_prefix} | Palette: {palette}', fig
    
    raise PreventUpdate



if __name__ == '__main__':
  app.run_server(debug=True)
  # app.enable_dev_tools(
  #   dev_tools_ui=True,
  #   dev_tools_serve_dev_bundles=True,
  # )
     