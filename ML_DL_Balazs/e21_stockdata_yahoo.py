import numpy as np
import  pandas_datareader.data as reader
import matplotlib.pyplot as plt

stock = ['AAPL']

start_date = '01/01/2001'
end_date = '01/01/2018'

data = reader.DataReader(sock)