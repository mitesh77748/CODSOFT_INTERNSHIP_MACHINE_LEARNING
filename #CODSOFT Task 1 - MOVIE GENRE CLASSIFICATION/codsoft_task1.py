import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

#importing data------------

train_data=pd.read_csv(r"E:\lab\project\codsoft\train_data.txt",sep=':::',names=['ID','TITLE','GENRE',"DESCRIPTION"])
print(train_data.head())
print(train_data.shape)

test_data=pd.read_csv(r"E:\lab\project\codsoft\test_data.txt",sep=':::',names=['ID','TITLE','GENRE','DESCRIPTION'])
print(test_data.head())
print(test_data.shape)

test_solution_data=pd.read_csv(r"E:\lab\project\codsoft\test_data_solution.txt",sep=':::',names=['ID','TITLE','GENRE','DESCRIPTION'])
print(test_solution_data.head())
print(test_solution_data.shape)

#Data visualization---------------------

plt.figure(figsize=(20,8))
sns.countplot(y=train_data['GENRE'],order=train_data['GENRE'].value_counts().index)
plt.title('Number of Movies per Genre')
plt.xlabel('No of movie per genere')
plt.ylabel('genere')
plt.show()

#bar representation of genere vs description length------------------

train_data['DESCRIPTION_length'] = train_data['DESCRIPTION'].apply(len)
plt.figure(figsize=(15, 10))
sns.barplot(x='GENRE', y='DESCRIPTION_length', data=train_data)
plt.title('Description Length by Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Description Length')
plt.show()

#genre which mostly people watched-----------------------

top_genres=train_data['GENRE'].value_counts().head(10)
plt.figure(figsize=(20,10))
top_genres.plot(kind='barh',color='cyan')
plt.title('Top 10 most frequent genres')
plt.xlabel('Number of movies')
plt.ylabel('genre')
plt.gca().invert_yaxis()
plt.show()

#training and testing of the data-----------------

train_data['DESCRIPTION'].fillna("",inplace=True)
test_data['DESCRIPTION'].fillna("",inplace=True)

t_v=TfidfVectorizer(stop_words='english',max_features=100000)
X_train=t_v.fit_transform(train_data['DESCRIPTION'])
X_test=t_v.transform(test_data['DESCRIPTION'])

label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(train_data['GENRE'])
y_test=label_encoder.transform(test_solution_data['GENRE'])

X_train_sub,X_val,y_train_sub,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

clf=LinearSVC()
clf.fit(X_train_sub,y_train_sub)
y_val_pred=clf.predict(X_val)
print("Validation Accuracy:",accuracy_score(y_val,y_val_pred))
print("validation Classification Report:\n",classification_report(y_val,y_val_pred))

y_pred=clf.predict(X_test)
print("Test Accuracy:",accuracy_score(y_test,y_pred))
print("Test Classification Report:\n",classification_report(y_test,y_pred))

from sklearn.naive_bayes import MultinomialNB
Mnb_classifier=MultinomialNB()
Mnb_classifier.fit(X_train,y_train)
Mnb_classifier.predict(X_test)

from sklearn.linear_model import LogisticRegression
lr_classifier=LogisticRegression(max_iter=500)
lr_classifier.fit(X_train,y_train)
lr_classifier.predict(X_test)


def predict_movie(description):
    t_v1=t_v.transform([description])
    pred_label=clf.predict(t_v1)
    return label_encoder.inverse_transform(pred_label)[0]
sample_descr_for_movie="Amovie where police cashes the criminal and shoot him"
print(predict_movie(sample_descr_for_movie))
sample_descr_for_movie1="A movie where person cashes a girl too get marry with him but girl refuses him"
print(predict_movie(sample_descr_for_movie1))