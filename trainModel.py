from sklearn.linear_model import Ridge
from error import MeanAverageError, MeanSquaredError, RootMeanSquaredError, R2score 
from print import printLinearRegressionModel, printError

def trainLR(X_train, X_test, y_train, y_test, regularization , errorType = "mse", printErr = 1, printModel = 1, isVal = 1):
    # Train the Ridge regression model
    ridge_model = Ridge(alpha=regularization)
    ridge_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = ridge_model.predict(X_train)
    y_pred_test = ridge_model.predict(X_test)
    
    
    
    if (errorType == "mse"):
        trainErr, testErr = MeanSquaredError(y_train, y_pred_train) , MeanSquaredError(y_test, y_pred_test)
    elif(errorType == "rmse"):
        trainErr, testErr = RootMeanSquaredError(y_train, y_pred_train) , RootMeanSquaredError(y_test, y_pred_test)
    elif(errorType == "mae"):
        trainErr, testErr = MeanAverageError(y_train, y_pred_train) , MeanAverageError(y_test, y_pred_test)
    elif(errorType == "r2"):
        trainErr, testErr = R2score(y_train, y_pred_train) , R2score(y_test, y_pred_test)

    #Print the Error
    if (printErr):
        printError(trainErr, testErr, errorType, isval = isVal)

    # Print the learned function
    if (printModel):
        printLinearRegressionModel(ridge_model.coef_, ridge_model.intercept_)
    
    return trainErr, testErr