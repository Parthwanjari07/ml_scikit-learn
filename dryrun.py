from urllib.request import urlretrieve
import plotly.express as px
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')
import pandas as pd
medical_df = pd.read_csv('medical.csv')
print(medical_df .sex.info())
print(medical_df.smoker.value_counts())