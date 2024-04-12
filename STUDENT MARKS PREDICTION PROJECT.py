
# create linear regression project.

# data spilitting.

import pandas as pd

# read files using pandas
 
dataset = pd.read_csv('02 students.csv')
df      = dataset.copy()

# split the variables vertically  into X and Y


X = df.iloc[:,:-1]       # select all coloums except last one.
Y = df.iloc[:, -1]       # select ony last one coloumn.


# split the dataset by rows

from sklearn.model_selection import train_test_split


x_train, x_test, y_train,y_test =    \
    train_test_split(X,Y, test_size = 0.3, random_state = 1234)

# create and train the multiple regression model.

from sklearn.linear_model import LinearRegression

# create Regressor 

std_reg = LinearRegression()

# train or fit the training data 

std_reg.fit(x_train,y_train)

# lets now predict the values of y from test data.
y_predict = std_reg.predict(x_test)

# calculate the R-squared and equation of line.

mlr_score = std_reg.score(x_test,y_test)

#coefficient of  the line 

mlr_coefficient = std_reg.coef_
mlr_intercept = std_reg.intercept_

# EQUATION OF LINE   

# Y= 1.31 + 4.67*hours + 5.1*shours

# HOW MUCH ERROR MODEL HAS MADE 
# RMSE - ROOT MEAN SQUARE ERROR

from sklearn.metrics import mean_squared_error
import math

mlr_rmse = math.sqrt(mean_squared_error(y_test,y_predict))

# PROJECT END 
