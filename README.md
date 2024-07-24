
# Student Marks Prediction Project
This project aims to predict students' marks based on various features using a Linear Regression machine learning model. The project involves data preprocessing, model training, and evaluation.

## Steps of project 
1. Dataset
2. Dependencies
3. Data preprocessing 
4. Model implementation 
5. Model training
6. Model Evaluation
7. Results
8. Conclusion

### Dataset
The dataset used in this project is stored in a CSV file named 02 students.csv. The dataset includes various features related to students and their marks.

### Dependencies
The project requires the following Python libraries:
* pandas
* scikit-learn

### Data preprocessing 
The dataset is loaded using the read_csv function from the pandas library. The features and target variable are then extracted into separate DataFrames.
``` python
import pandas as pd
dataset = pd.read_csv('02 students.csv')
df = dataset.copy()

# Data Splitting
# The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]  # Select all columns except the last one
Y = df.iloc[:, -1]   # Select only the last column
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
```
### Model implementation
``` python 
 # Linear Regression Model
 # A Linear Regression model is created and trained on the training data. Predictions are then made on the test data.

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
```
### Model training
``` python
std_reg.fit(x_train, y_train)
y_predict = std_reg.predict(x_test)
```
### Model Evaluation
``` python
The model is evaluated using the R-squared score and Root Mean Squared Error (RMSE).

from sklearn.metrics import mean_squared_error
import math

mlr_score = std_reg.score(x_test, y_test)
mlr_coefficient = std_reg.coef_
mlr_intercept = std_reg.intercept_
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
```
### Results
``` python
R-squared Score: mlr_score
Coefficients of the Line: mlr_coefficient
Intercept of the Line: mlr_intercept
Root Mean Squared Error (RMSE): mlr_rmse
```
![Screenshot 2024-07-18 173146](https://github.com/user-attachments/assets/0736b96e-ae8b-404a-acca-fa00cc62b353)


### Conclusion
This project demonstrates the use of Linear Regression for predicting students' marks. The results indicate the model's performance in terms of R-squared score and RMSE. Future work can include testing other regression models and feature selection techniques to further improve prediction accuracy.




