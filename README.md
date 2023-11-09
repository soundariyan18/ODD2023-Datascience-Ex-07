# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## EXPLANATION
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## PROGRAM

```
Developed by: SOUNDARIYAN M.N
REG: 212222230146

```


```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns

df.shape

x=df.drop("Survived",1)
y=df['Survived']

df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns

df1['Age'].isnull().sum()

df1['Age'].fillna(method='ffill')

df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()

feature=SelectKBest(mutual_info_classif,k=3)

df1.columns

cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]

df1.columns

x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns

y=y.to_frame()

y.columns

from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

x

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes

data

k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x,y)

selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)

selected_features = x.columns[rfe.support_]

print("Selected Features:")
print(selected_feature)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x,y)

feature_importances = model.feature_importances_

threshold = 0.15

selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```
## OUPUT

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/3fe4d229-7f3f-4f94-b194-c0949ca6dd98)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/635fa9c5-f45d-4432-bd92-18bfd79adaa0)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/cc613fca-9cc5-4af5-acb5-619fa21bcc80)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/d9e593cd-cf97-4a23-9abd-4ff863c9fc68)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/60da11b1-4fba-4a9c-88a6-c88f7f4f9905)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/0aaf79df-689d-458e-8428-fdc5324ad3ce)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/4843dc61-20d6-4c2b-a6b5-0e1c07f92ca6)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/3f399b26-da54-4104-95ea-d2813911c241)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/c53dd8bd-f454-4178-b07a-eae71c2adc5a)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/0488bcfc-aeb4-4103-ab1e-83c8d0911cac)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/d3102182-7dcd-4ba4-a8ee-c1bdd3a71a73)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/55f5a2e6-7fe9-412e-a2a6-f6d5d5778259)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/a44677a3-17d8-4b50-bb77-9bd1ac5196f4)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/c99f978c-636b-413b-a3bb-2a6a7dc97121)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/a0b3e650-c003-43e0-9c0c-7db2073a8660)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/d7d9fbe3-208f-4b1e-b8b6-f640b8d3e0ab)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/e388a913-01a0-4d08-a981-400ea8916a27)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/42eb1bba-deea-48e7-a8af-88bdf045a0b0)

![image](https://github.com/soundariyan18/ODD2023-Datascience-Ex-07/assets/119393307/4035f027-34f0-4c49-94ba-935a2a393f2c)

## RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully

