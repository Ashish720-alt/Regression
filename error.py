
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
import math

#Use MSE or RMSE when want to penalize outliers heavily
def MeanSquaredError(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

def RootMeanSquaredError(y_test, y_pred):
    return math.sqrt(mean_squared_error(y_test, y_pred))

#Use MAE when don't want to penalize outliers heavily
def MeanAverageError(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

#Use R2 score a.k.a. coefficient of determination alongside MAE and MSE to help measure variance of error
def R2score(y_test, y_pred):
    return r2_score(y_test, y_pred)


def isErrorHigh(error, baselineError, tolerance = 1.1):
    return (error > tolerance * baselineError)