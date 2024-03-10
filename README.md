# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize theta
2. Calculate predictions for all
3. Calculate the error for all
4. Calculate the gradients (derivatives)
5. Update parameters:
            - theta = theta- learning_rate * gradient_theta
6. Return theta0 and theta1

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: YOGARAJ . S
RegisterNumber:  212223040248
*/
```
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):

  x = np.c_[np.ones(len(x1)),x1]
  
  theta = np.zeros(x.shape[1]).reshape(-1,1)
  
  for _ in range(num_iters):
  
    predictions =(x).dot(theta).reshape(-1,1)
    
    errors = (predictions - y).reshape(-1,1)
    
    theta -= learning_rate *(1 / len(x1)) * x.T.dot(errors)

  return theta

data = pd.read_csv('/content/50_Startups.csv',header=None)

X = (data.iloc[1:, :-2].values)

X1 = X.astype(float)

scaler = StandardScaler()

y = (data.iloc[1:,-1].values).reshape(-1,1)

X1_Scaled = scaler.fit_transform(X1)

Y1_Scaled = scaler.fit_transform(y)

theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)

new_Scaled = scaler.fit_transform(new_data)

prediction = np.dot(np.append(1, new_Scaled), theta)

prediction = prediction.reshape(-1,1)

pre = scaler.inverse_transform(prediction)

print (f"Predicted value: {pre}")

## Output:
![linear regression using gradient descent](sam.png)
![image](https://github.com/yogaraj2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/153482637/fbad4c6a-e2ff-41ab-9c0e-877848bec0fa)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
