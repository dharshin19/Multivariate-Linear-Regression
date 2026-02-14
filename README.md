# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
### Step1
import pandas as pd.
### Step2
Read the csv file.

### Step3
Get the value of X and y variables

### Step4
Create the linear regression model and fit.

### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.


## Program:
```


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston = pd.read_csv(url)

X=boston.drop("medv", axis=1)
y=boston["medv"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
print("coefficient:",reg.coef_)
print("Variance score: {}",format(reg.score(X_test,y_test)))
plt.style.use("fivethirtyeight")
plt.scatter(reg.predict(X_train), reg.predict(X_train)-y_train,color="green",s=10,label="Train data")
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,color="blue",s=10,label="Test data")
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)
plt.legend(loc="upper right")
plt.title("Residuals errors")
plt.show()



```
## Output:

### Insert your output

<img width="1910" height="1637" alt="screencapture-localhost-8888-notebooks-Untitled13-ipynb-2026-02-14-08_39_07" src="https://github.com/user-attachments/assets/00602f13-0eb0-49da-ae88-a94e3f6e86e3" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
