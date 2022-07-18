# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

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


def LR_Model(x, iterations):   

    train_acc = []
    test_acc = [] 
    logLossData = []
    random = range(iterations)
    for i in random:
        #Split Data
        x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.30,  stratify = y, random_state=i)    
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)
    
        #Logistic Regression
        logReg=LogisticRegression().fit(x_train,y_train)
        yPred_train = logReg.predict(x_train)
        yPred_test = logReg.predict(x_test)
    
        train_acc.append(accuracy_score(y_train, yPred_train).round(2))
        test_acc.append(accuracy_score(y_test, yPred_test).round(2))
    
        logLoss = log_loss(y_test,yPred_test)
        print("LogLoss value:", logLoss)
        logLossData.append(logLoss)
        performance_measure(y_test, yPred_test)
        coeff = list(logReg.coef_[0])
        labels = list(x_train.columns)
        features = pd.DataFrame()
        features['Features'] = labels
        features['importance'] = coeff
        features.sort_values(by=['importance'], ascending=True, inplace=True)
        features['positive'] = features['importance'] > 0
        features.set_index('Features', inplace=True)
        features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
        plt.xlabel('Importance')
        roc_auc(y_test,yPred_test)
        
    print(train_acc)
    print(test_acc)
    print(logLossData) 
   
    meanTestAccuracy = np.mean(test_acc)
    stdDev = np.std(test_acc)    
    minValue =(meanTestAccuracy- (3*stdDev));
    maxValue = (meanTestAccuracy+ (3*stdDev));
       
    print("Minimun Value:", minValue);
    print("Maximum Value:", maxValue);
    print("Average Accuracy:", meanTestAccuracy);
    print("LogLossAverage:", np.mean(logLossData))
    testaccuracies = pd.DataFrame(test_acc)
    testaccuracies.plot.hist()  
    
    return meanTestAccuracy.round(2)

def knn_model(x, iterations):
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=40, stratify=y)
    test_acc =[]
    random = range(1,iterations)
    for i in random:
        knn=KNeighborsClassifier(n_neighbors=i, p=2)    
        knn.fit(x_train,y_train)
        y_pred_test=knn.predict(x_test)
        test_acc.append(accuracy_score(y_test,y_pred_test).round(2))
        performance_measure(y_test,y_pred_test)
        roc_auc(y_test,y_pred_test)
    print(test_acc)
    
    plt.plot(random, test_acc)
    plt.ylabel('Accuracies')
    return test_acc[0],test_acc[1],test_acc[2]
    
def sgd_model(x, iterations):
    test_acc = []
    for i in range(iterations):
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=i, stratify=y)
        sgd = SGDClassifier(loss='hinge')
        sgd.fit(x_train,y_train)
        y_pred = sgd.predict(x_test)
        sgd.intercept_
        sgd.coef_
        performance_measure(y_test,y_pred)
        roc_auc(y_test,y_pred)
        test_acc.append(accuracy_score(y_pred, y_test).round(2))
    print(np.mean(test_acc))    
    return (np.mean(test_acc)).round(2)

def performance_measure(y_test,yPred_test):
    cm = confusion_matrix(y_test, yPred_test)
    print(cm)
    sensitivity = cm[1,1]/(cm[1,1]+cm[0,1])
    specificity = cm[0,0]/(cm[0,0]+cm[1,0])
    precision = precision_score(yPred_test, y_test)
    f1Score = f1_score(yPred_test, y_test)
    print("Sensitivity score is: ", sensitivity.round(2))
    print("Specificity score is: ", specificity.round(2))
    print("Precision is: ", precision.round(2)) 
    print("f1 score is : ", f1Score.round(2))

def roc_auc(y_test,yPred_test):
    fpr, tpr, thresholds = roc_curve(y_test, yPred_test)
    auc = roc_auc_score(y_test, yPred_test)
    print("roc_auc_score:",auc)
    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
scaled_df = scaleData(df.columns[4:14])
y = df['Gender_encoded'] 
x= scaled_df.loc[:, ['inter canine distance intraoral', 
                     'intercanine distance casts', 
                     'right canine width intraoral',
                     'right canine width casts',
                     'left canine width intraoral',
                     'left canine width casts',
                     'right canine index intra oral',
                     'right canine index casts',
                      'left canine index intraoral',
                     'left canine index casts'
                      ]]
LR_Model(x,1)
knn_model(x,5)
sgd_model(x,1)

#Model Building and feature selection
lr_acc = []
knn_acc =[]
sgd_acc = []


features = [scaled_df.loc[:, ['inter canine distance intraoral',
                              'left canine width casts']],
            scaled_df.loc[:, ['inter canine distance intraoral', 
                              'left canine width casts',
                              'left canine index casts']],
            scaled_df.loc[:, ['inter canine distance intraoral',
                              'left canine width casts',
                              'left canine index intraoral']],
            scaled_df.loc[:, ['inter canine distance intraoral',
                              'left canine width intraoral']],
             scaled_df.loc[:, ['left canine width casts',
                              'right canine width casts',
                              'left canine index casts'
                              ]],
            scaled_df.loc[:, ['left canine width intraoral',
                              'right canine width intraoral',
                              'left canine index intraoral'
                              ]]
            ]
for i in features:
    lr_acc.append(LR_Model(i,1))
    knn_acc.append(knn_model(i,5))
    sgd_acc.append(sgd_model(i,1))
    
#Cross Validation
for i in features:
    lr_acc.append(LR_Model(i,100))
for i in features:    
    knn_acc.append(knn_model(i,20))
for i in features:
    sgd_acc.append(sgd_model(i,100))
    
#P-Values    
for i in range(len(features)):
    xNew = sm.add_constant(features[i])
    lm2=sm.OLS(y,xNew).fit()
    print("P Values for model with selected features = ", list(features[i]))
    print(lm2.summary())
    
# evaluate accuracies
for i in range(len(features)):
    print(list(features[i]), lr_acc[i],knn_acc[i],sgd_acc[i])






