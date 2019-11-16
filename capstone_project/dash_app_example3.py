
import dash
import datetime
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas_datareader.data as web

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2018,2,8)

df = web.DataReader('TSLA','google',start,end)
print(df.head())