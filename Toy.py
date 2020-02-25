# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:11:44 2019

@author: user

It is a Toy example for continual learning for time series
use the air quality dataset (https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
"""

dataPath = "/home/user/code/RNN/datasets/PRSA_data_2010.1.1-2014.12.31.csv"


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(dataPath)
plt.plot(df["PRES"])




