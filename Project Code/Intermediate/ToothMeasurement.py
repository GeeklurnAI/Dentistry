# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.expand_frame_repr', False)


df = pd.read_csv("Proj1.csv")

#exploring data
print(df)
print(df.shape)
print(df.dtypes)
print(list(df))
print(df.info())
print(df.isnull())
#Label Encoding
df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])
print(df['Gender'].value_counts())
df_dummy = df.rename(columns = {'inter canine distance intraoral':'icd_oral', 
                     'intercanine distance casts':'icd_casts',
                     'right canine width intraoral': 'rcw_oral',
                     'right canine width casts': 'rcw_casts',
                     'left canine width intraoral':'lcw_oral',
                     'left canine width casts':'lcw_casts',
                     'right canine index intra oral':'rci_oral',
                     'right canine index casts': 'rci_casts',
                      'left canine index intraoral':'lci_oral',
                      'left canine index casts': 'lci_casts'}, inplace = False)
print(list(df_dummy))

#Visualisations
sns.countplot(x='Gender',data=df,palette = 'hls')
plt.show()
plt.savefig('count_plot')
#data is balanced here

#Correlation table
corr = df[df.columns[4:15]].corr()
print(df_dummy[df_dummy.columns[4:15]].corr())
sns.heatmap(corr, 
         xticklabels=corr.columns, 
         yticklabels=corr.columns)

#Outlier visualisation
outlier = []
def outliers(col, df):
    df[col].quantile([.25, .50, .75])

    Q1 = np.percentile(df[col],25) # Compute percentile vlaues
    Q2 = np.percentile(df[col],50)
    Q3 = np.percentile(df[col],75)
    
    print("Quartile 1 :", Q1)
    print("Quartile 2 :", Q2)
    print("Quartile 3 :", Q3)
    
    IQR = Q3 -Q1
    print("Inter Quartile Range :", IQR)
    
    Lower_whisker = Q1 - (1.5 * IQR)
    print("Lower Whisker value :", Lower_whisker)
    
    Upper_whisker = Q3 + (1.5 * IQR)
    print("Upper Whisker value :", Upper_whisker)    
    
    df[df[col] < Lower_whisker]
    df[df[col] > Upper_whisker]    
    
    outlier.append([col, len(df[(df[col] < Lower_whisker) | (df[col] > Upper_whisker)])]) 
    print('Total number of outliers')
    print(len(df[(df[col] < Lower_whisker) | (df[col] > Upper_whisker)]))
    return [Lower_whisker, Upper_whisker]



for col in df.columns[4:14]:
    plt.boxplot(df[col])
    plt.xlabel(col)
    plt.show() 
    outliers(col,df)
print ('Outliers')
for i in range(len(outlier)):
        print(str(outlier[i][0]) + ' - ' + str(outlier[i][1]))
 
#Data PreProcessing 
def scaleData(columns):    
    scaledValues = MinMaxScaler().fit_transform(df.loc[:, columns])
    print(scaledValues)
    scaled_df = pd.DataFrame(scaledValues)
    scaled_df.shape
    scaled_df.columns = columns
    list(scaled_df)
    return scaled_df

scaled_df = scaleData(df.columns[4:14])