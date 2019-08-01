import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
##1:LOAD DATASET
dataframe=pd.read_csv("spam.csv")
print(dataframe.head())

##2:train
x=dataframe["EmailText"]
y=dataframe["Label"]

x_train,y_train=x[0:4457],y[0:4457]
x_test,y_test=x[4457:],y[4457:]

##3:extract features
cv= CountVectorizer()
features=cv.fit_transform(x_train)
##4:build a model
model=svm.SVC()
model.fit(features,y_train)
##5 test accuracy
features_test=cv.transform(x_test)
print("Accuracy of the model is",model.score(features_test,y_test))


