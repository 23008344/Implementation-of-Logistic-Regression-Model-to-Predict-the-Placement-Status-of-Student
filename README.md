# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARNIKA.P
RegisterNumber: 212223240170 
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Y_prediction array:
![image](https://github.com/23008344/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742655/76f0f46b-a65b-4532-b78e-d7ed3910d782)

## Accuracy value:
![image](https://github.com/23008344/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742655/1c1ae09e-1d15-457c-be7a-c4df6018d903)

## Confusion array:
![image](https://github.com/23008344/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742655/cdc99b2a-266f-4e39-98a1-28a073809b1d)

## Classification Report:
![image](https://github.com/23008344/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742655/9db6e8bd-3b1a-4a1a-bb3e-92dbb89e1039)

## Prediction of LR:
![image](https://github.com/23008344/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742655/a8b66007-dff9-4386-947a-35911d3b3209)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
