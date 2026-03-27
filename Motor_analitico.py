"""
Importamos las librerías necesarias
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandasql import sqldf

#Dataframe para 1_Data_Recordings.csv
df_1_Data_Recordings = pd.read_csv('1_Data_Recordings.csv')

#Dataframe para 2_Data_Recordings.csv
df_2_Data_Recordings = pd.read_csv('2_Data_Recordings.csv')

#Ejecutamos sql
pysqldf = lambda q: sqldf(q, globals())

