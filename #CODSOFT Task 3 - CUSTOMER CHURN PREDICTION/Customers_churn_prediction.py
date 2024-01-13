import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r'E:\lab\project\Machine_learning\codsoft_task 3\Churn_Modelling.csv')
print(data)
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.columns)

data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
print(data)

data=pd.get_dummies(data,drop_first=True)
print(data.head())
data=data.astype(int)
print(data)
print(data['Exited'].value_counts())

plt.figure(figsize =(8,6))
sns.countplot(x='Exited',data = data)
plt.show()

x=data.drop('Exited',axis=1)
y=data['Exited']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler,LabelEncoder

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
print("Training shape:",X_train.shape)
print("Testing Shape:",X_test)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)

threshold=0.5
y_tarin_classified=[1 if value > threshold else 0 for value in y_train]
LR=LogisticRegression()
LR.fit(X_train_scaled,y_tarin_classified)

y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy1=LR.score(X_test_scaled,y_test_classified)
print('Model Accuracy:',accuracy1)

from sklearn import svm
threshold=0.5
y_train_classified=[1 if value > threshold else 0 for value in y_train]
svm=svm.SVC()
svm.fit(X_train_scaled,y_train_classified)
y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy2=svm.score(X_test_scaled,y_test_classified)
print('Model Accuracy:',accuracy2)

threshold=0.5
y_train_classified=[1 if value > threshold else 0 for value in y_train]
rf=RandomForestClassifier()
rf.fit(X_train_scaled,y_train_classified)
y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy3=rf.score(X_test_scaled,y_test_classified)
print("Model Accuracy:",accuracy3)

threshold=0.5
y_train_classified=[1 if value > threshold else 0 for value in y_train]
dt=DecisionTreeClassifier()
dt.fit(X_train_scaled,y_train_classified)
y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy4=dt.score(X_test_scaled,y_test_classified)
print("Model Accuracy:",accuracy4)

threshold=0.5
y_train_classified=[1 if value > threshold else 0 for value in y_train]
KNN=KNeighborsClassifier()
KNN.fit(X_train_scaled,y_train_classified)
y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy5=KNN.score(X_test_scaled,y_test_classified)
print("Model Accuracy:",accuracy5)

from sklearn.ensemble import GradientBoostingClassifier
threshold=0.5
y_train_classified=[1 if value > threshold else 0 for value in y_train]
GBC=GradientBoostingClassifier()
GBC.fit(X_train_scaled,y_train_classified)
y_test_classified=[1 if value > threshold else 0 for value in y_test]
accuracy6=GBC.score(X_test_scaled,y_test_classified)
print("Model Accuracy:",accuracy6)

performance_summary=pd.DataFrame({
'Model':['LR','svm','KNN','dt','rf','GBC'],
'ACC':[accuracy1,
       accuracy2,
       accuracy3,
       accuracy4,
       accuracy5,
       accuracy6
       ]

})
print(performance_summary)