from urllib.request import urlretrieve
import plotly.express as px
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression




sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')
import pandas as pd
medical_df = pd.read_csv('medical.csv')
#print(medical_df .sex.describe())
#print(medical_df.smoker.value_counts())

fig_age = px.histogram( medical_df,
                   x= 'age',
                   marginal='box',
                   nbins= 47,
                   title = 'Diatribution of Age')
fig_age.update_layout(bargap= 0.1)
#fig_age.show()

fig_bmi = px.histogram(medical_df,
                   x='bmi',
                   marginal="box",
                   color_discrete_sequence=['red'],
                   title=("Distribution pf BMI"))

fig_bmi.update_layout(bargap =0.2)
#fig_bmi.show()

fig_charges = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig_charges.update_layout(bargap=0.1)
#fig_charges.show()

fig_sex = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='sex', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig_sex.update_layout(bargap=0.1)
#fig_sex.show()

fig_smokerxsex = px.histogram(medical_df,
                   x= 'smoker',
                   color ='sex')

#fig_smokerxsex.show()

fig_agexcharges = px.scatter(medical_df, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig_agexcharges.update_traces(marker_size=7)
#fig_agexcharges.show()

fig_bmixcharges = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Bmi vs. Charges')
fig_bmixcharges.update_traces(marker_size=7)
#fig_bmixcharges.show()

fig_children= px.violin(medical_df, x='children', y='charges')
#fig_children.show()

#correlation between columns
#print(medical_df.charges.corr(medical_df.age))
#print(medical_df.charges.corr(medical_df.bmi))


# .map maps the user defined value on the already present value 
smoker_values = {'no': 0, 'yes': 1}
medical_df['smoker_numeric'] = medical_df['smoker'].map(smoker_values)
#print(smoker_numeric)
#print(medical_df.charges.corr(smoker_numeric))

sex_values = {'male': 1, 'female': 0}
medical_df['sex_numeric'] = medical_df['sex'].map(sex_values)



#dropping non numeric columns

numeric_columns = medical_df.select_dtypes(include = ['float64', 'int64']).columns
correlation_matrix = medical_df[numeric_columns].corr()




#sns.heatmap(correlation_matrix.corr(), cmap='Reds', annot=True, cbar_kws={'shrink': 0.8}, square= True)
#plt.title('Correlation Matrix')
#plt.show()

#starting linear regression

non_smoker_df = medical_df[medical_df.smoker == 'no']

#plt.title('age vs Charges')
#sns.scatterplot(data= non_smoker_df,x='age',y= 'charges', alpha =0.7, s= 15)
#plt.show()

#plt.title('age vs Charges')
#sns.scatterplot(data= non_smoker_df,x='age',y= 'charges', alpha =0.7, s= 15)
#plt.show()

def estimate_charges(age,w,b):
    return w* age + b


#calculating error RMSE(root mean square error)
def rmse(targets,predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

#function to try paramater 

def try_parameters(w,b):
    ages =non_smoker_df.age
    target =non_smoker_df.charges
    #print(ages)
    Estimate_charges=estimate_charges(ages, w,b)
    #print(Estimate_charges)

    
    

    plt.plot(ages, Estimate_charges, 'r-', alpha  = 0.9)
    plt.scatter(ages,target,s=8, alpha=0.7)
    plt.xlabel('Age')
    plt.ylabel('charges')
    
    error =rmse(target, Estimate_charges)
    print(f"RMSE :{error}")
    plt.show()

#try_parameters(275,-3500)


#prediction using single feature linear regression
    
targets= non_smoker_df.charges

def prediction_single_feature():
    model =LinearRegression()

    #help(model.fit)

    inputs = non_smoker_df[['age']]
    
    #print('input.shape:',inputs.shape)
    #print('targets.shape:',targets.shape)

    model.fit(inputs,targets)
    predictions =model.predict(inputs)

    print(predictions)
    #print(targets)

    error = rmse(targets, predictions)
    print('RMSE:',error)

    #print(model.coef_)
    #print(model.intercept_)



#linear regression witth multiple features
def prediction_multi_feature():
    inputs_multi = non_smoker_df[['age','bmi']]

    model_multi = LinearRegression().fit(inputs_multi, targets)

    predictions_multi = model_multi.predict(inputs_multi)
    print(predictions_multi) 

    error_multi = rmse(targets, predictions_multi)
    print('RMSE:',error_multi)


prediction_single_feature()
prediction_multi_feature()



